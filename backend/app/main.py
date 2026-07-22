"""Main FastAPI application module for the puzzle solver."""

import asyncio
import base64
import io
import logging
import os
import uuid
from typing import Annotated, List, Literal, Optional

import numpy as np
from fastapi import Depends, FastAPI, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from app.auth.dependencies import get_current_user, get_owned_puzzle
from app.auth.service import AuthService, get_auth_service
from app.config import settings
from app.models.piece_geometry_model import (
    GeometryEdge,
    GeometryPoint,
    GeometryQuality,
    PieceGeometryListResponse,
    PieceGeometryRecordResponse,
    PieceGeometrySummary,
    PieceGeometryUploadResponse,
)
from app.models.puzzle_model import (
    BarcodeLookupResponse,
    BoundingBox,
    Corner,
    CutPuzzleRequest,
    CutPuzzleResponse,
    DetectFrameResponse,
    GeneratePieceRequest,
    GeneratePieceResponse,
    PiecePreviewResponse,
    PieceResponse,
    PuzzleListResponse,
    PuzzleResponse,
    PuzzleSummary,
    QuadCorners,
)
from app.models.user_model import GoogleAuthRequest, TokenResponse, User
from app.rate_limit import FixedWindowRateLimiter, rate_limit_by_ip, rate_limit_by_user
from app.services.barcode_lookup_cache import BarcodeLookupRecord, get_barcode_lookup_cache
from app.services.classical_matcher import get_classical_matcher
from app.services.grid_snap import snap_to_grid
from app.services.image_processor import get_image_processor
from app.services.piece_detector import get_piece_detector
from app.services.piece_geometry.service import (
    PieceGeometryRecord,
    get_piece_geometry_service,
    quick_quality_from_polygon,
)
from app.services.piece_geometry.store import get_piece_geometry_store
from app.services.piece_shape import PieceShapeGenerator, calculate_grid_dimensions
from app.services.puzzle_cutter import get_puzzle_cutter
from app.services.puzzle_detector import get_puzzle_detector
from app.services.puzzle_store import PuzzleRecord, get_puzzle_store
from app.services.ravensburger_client import get_ravensburger_client
from app.services.ravensburger_lookup import RAVENSBURGER_ADULT_PREFIX, candidate_article_numbers, ean_checksum_valid

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=settings.PROJECT_NAME)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# piece_count must produce a sane grid: a puzzle can't reasonably have fewer
# than 4 pieces, and this backend isn't tuned for four-digit piece counts.
MIN_PIECE_COUNT = 4
MAX_PIECE_COUNT = 2000

# Rate limiters (see app/rate_limit.py). One instance per protected route so
# their counters don't collide; limits are read live from settings on each
# request (see app/config.py: RATE_LIMIT_AUTH_PER_MINUTE / _PREVIEW_PER_MINUTE).
auth_rate_limiter = FixedWindowRateLimiter()
piece_preview_rate_limiter = FixedWindowRateLimiter()
barcode_lookup_rate_limiter = FixedWindowRateLimiter()


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# =============================================================================
# Authentication Endpoints
# =============================================================================


@app.post(
    "/api/v1/auth/google",
    response_model=TokenResponse,
    dependencies=[Depends(rate_limit_by_ip(auth_rate_limiter, lambda: settings.RATE_LIMIT_AUTH_PER_MINUTE))],
)
async def google_auth(
    request: GoogleAuthRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Authenticate with Google OAuth.

    Exchanges a Google ID token for an application JWT token.

    Unauthenticated and rate-limited per-IP (see `app.rate_limit`): this
    endpoint does a network call to Google's cert endpoint per request, so
    without a limit it's both brute-forceable and a free DoS amplifier.

    Args:
        request: The Google authentication request containing the ID token.
        auth_service: The authentication service.

    Returns:
        TokenResponse containing the access token and user information.

    Raises:
        HTTPException: If the Google token is invalid, the account's email
            isn't on the configured allowlist (see `Settings.ALLOWED_EMAILS`),
            or the caller's IP has exceeded `RATE_LIMIT_AUTH_PER_MINUTE` (429).
    """
    user_info = auth_service.verify_google_token(request.id_token)
    if user_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid Google token",
        )

    if not settings.is_email_allowed(user_info["email"]):
        raise HTTPException(
            status_code=403,
            detail="This account is not authorized to use this application",
        )

    user = User(
        id=user_info["sub"],
        email=user_info["email"],
        name=user_info["name"],
        picture=user_info.get("picture"),
    )

    access_token, expires_in = auth_service.create_access_token(user)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@app.get("/api/v1/auth/me", response_model=User)
async def get_current_user_profile(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get the current authenticated user's profile.

    Args:
        current_user: The authenticated user from the JWT token.

    Returns:
        The current user's profile information.
    """
    return current_user


# =============================================================================
# Puzzle Endpoints (Protected)
# =============================================================================


@app.post("/api/v1/puzzle/upload", response_model=PuzzleResponse)
async def upload_puzzle(
    current_user: Annotated[User, Depends(get_current_user)],
    file: Optional[UploadFile] = None,
    piece_count: Annotated[
        Optional[int],
        Form(description="Total number of pieces in the puzzle; used to estimate the grid (rows x cols)"),
    ] = None,
) -> PuzzleResponse:
    """Upload a complete puzzle image.

    Args:
        current_user: The authenticated user.
        file: The puzzle image file.
        piece_count: Optional total piece count supplied by the client (e.g. the
            iOS app). When provided, the grid is estimated from this count and
            the image's aspect ratio and stored alongside the puzzle.

    Returns:
        PuzzleResponse: Response containing the puzzle ID, and — when
        piece_count was provided — the echoed piece_count and estimated
        rows/cols.

    Raises:
        HTTPException: If file size exceeds limit, piece_count is out of
            range, or (when piece_count is given) the file isn't a decodable
            image.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    if piece_count is not None and not (MIN_PIECE_COUNT <= piece_count <= MAX_PIECE_COUNT):
        raise HTTPException(
            status_code=400,
            detail=f"piece_count must be between {MIN_PIECE_COUNT} and {MAX_PIECE_COUNT}",
        )

    contents = await file.read()

    rows: Optional[int] = None
    cols: Optional[int] = None
    if piece_count is not None:
        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid image file") from exc
        rows, cols = calculate_grid_dimensions(image.width, image.height, piece_count)

    puzzle_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{puzzle_id}.jpg")

    with open(file_path, "wb") as f:
        f.write(contents)

    get_puzzle_store().add(
        PuzzleRecord(
            puzzle_id=puzzle_id,
            owner_id=current_user.id,
            file_path=file_path,
            grid=(rows, cols) if rows is not None and cols is not None else None,
        )
    )

    return PuzzleResponse(puzzle_id=puzzle_id, piece_count=piece_count, rows=rows, cols=cols)


@app.get("/api/v1/puzzles", response_model=PuzzleListResponse)
async def list_puzzles(
    current_user: Annotated[User, Depends(get_current_user)],
) -> PuzzleListResponse:
    """List the caller's own puzzles.

    Args:
        current_user: The authenticated user.

    Returns:
        PuzzleListResponse with one summary (puzzle_id, plus rows/cols when
        known) per puzzle owned by the caller, in upload order. Never
        includes other users' puzzles.
    """
    records = get_puzzle_store().list_for_owner(current_user.id)
    return PuzzleListResponse(
        puzzles=[
            PuzzleSummary(
                puzzle_id=record.puzzle_id,
                rows=record.grid[0] if record.grid is not None else None,
                cols=record.grid[1] if record.grid is not None else None,
            )
            for record in records
        ]
    )


@app.post(
    "/api/v1/puzzle/detect-frame",
    response_model=DetectFrameResponse,
    dependencies=[Depends(get_current_user)],
)
async def detect_frame(
    file: Optional[UploadFile] = None,
    corners: Annotated[Optional[str], Form(description="Manually adjusted corners as JSON QuadCorners")] = None,
) -> DetectFrameResponse:
    """Detect the puzzle picture in a photo and return a perspective-corrected crop.

    This endpoint is stateless: nothing is persisted. The frontend previews the
    trimmed image and, once accepted, uploads it via the regular upload endpoint.
    When ``corners`` is provided (manual adjustment), detection is skipped and
    the supplied corners are used for the warp with confidence 1.0.

    Requires authentication (see the route's ``dependencies``), but doesn't
    need the caller's identity — it doesn't persist or look up anything.

    Args:
        file: The raw photo of the puzzle.
        corners: Optional JSON-encoded QuadCorners overriding auto-detection.

    Returns:
        DetectFrameResponse with the trimmed image (base64 data URL), the
        corners used, and a detection confidence.

    Raises:
        HTTPException: If no/invalid file is provided, the file is too large,
            or the corners payload is malformed.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        image = Image.open(io.BytesIO(contents))
        # Phone photos carry EXIF orientation; correct it so corners match what the user sees
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    detector = get_puzzle_detector()
    if corners is not None:
        try:
            quad = QuadCorners.model_validate_json(corners)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="Invalid corners payload") from exc
        used_corners = quad
        confidence = 1.0
        trimmed = detector.warp(image, quad.as_points())
    else:
        points, confidence = detector.detect_corners(image)
        used_corners = QuadCorners.from_points(points)
        trimmed = detector.warp(image, points)

    buffer = io.BytesIO()
    trimmed.save(buffer, format="JPEG", quality=90)
    trimmed_base64 = base64.b64encode(buffer.getvalue()).decode()

    return DetectFrameResponse(
        trimmed_image=f"data:image/jpeg;base64,{trimmed_base64}",
        corners=used_corners,
        confidence=confidence,
    )


@app.get(
    "/api/v1/puzzle/barcode/{ean}",
    response_model=BarcodeLookupResponse,
    dependencies=[
        Depends(rate_limit_by_user(barcode_lookup_rate_limiter, lambda: settings.RATE_LIMIT_BARCODE_PER_MINUTE))
    ],
)
async def lookup_barcode(ean: str) -> BarcodeLookupResponse:
    """Look up a Ravensburger box image by EAN-13 barcode.

    Stateless from the caller's perspective: derives candidate article
    numbers from the EAN (see `app.services.ravensburger_lookup`), probes
    the Ravensburger image CDN concurrently, and returns the first real
    image (converted webp -> JPEG data URL). Results — hits and misses —
    are cached in-process so a barcode held in front of the camera doesn't
    re-probe the CDN.

    Requires authentication and is rate-limited per-user (see the route's
    ``dependencies``); doesn't need the caller's identity beyond that.

    Args:
        ean: The scanned EAN-13 barcode payload.

    Returns:
        BarcodeLookupResponse with found=True plus the box image and
        resolved article number, or found=False when the EAN isn't a
        (known) Ravensburger product.

    Raises:
        HTTPException: If the EAN is malformed (400), or the caller has
            exceeded `RATE_LIMIT_BARCODE_PER_MINUTE` (429).
    """
    if not ean_checksum_valid(ean):
        raise HTTPException(status_code=400, detail="Invalid EAN-13 barcode")

    cache = get_barcode_lookup_cache()
    cached = cache.get(ean)
    if cached is None:
        cached = await _resolve_barcode(ean)
        cache.put(ean, cached)

    if not cached.found or cached.box_image_jpeg is None:
        return BarcodeLookupResponse(found=False)
    box_image_base64 = base64.b64encode(cached.box_image_jpeg).decode()
    return BarcodeLookupResponse(
        found=True,
        box_image=f"data:image/jpeg;base64,{box_image_base64}",
        article_number=cached.article_number,
    )


async def _resolve_barcode(ean: str) -> BarcodeLookupRecord:
    """Probe the Ravensburger CDN for a valid EAN and build a cacheable record.

    Args:
        ean: A checksum-valid EAN-13.

    Returns:
        A BarcodeLookupRecord holding the JPEG-converted box image of the
        first candidate article number that hit (in preference order), or a
        miss record when no candidate produced a real image.
    """
    candidates = candidate_article_numbers(ean, settings.RAVENSBURGER_SERIES_PREFIXES)
    if not candidates:
        return BarcodeLookupRecord(found=False, box_image_jpeg=None, article_number=None)

    client = get_ravensburger_client()
    results = await asyncio.gather(*(client.fetch_box_image(candidate) for candidate in candidates))
    for candidate, image_bytes in zip(candidates, results):
        if image_bytes is None:
            continue
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            logger.warning("Ravensburger CDN returned an undecodable image for article %s", candidate)
            continue
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return BarcodeLookupRecord(found=True, box_image_jpeg=buffer.getvalue(), article_number=candidate)

    if ean.startswith(RAVENSBURGER_ADULT_PREFIX):
        # Adult-line article numbers aren't fully derivable from the EAN; log
        # all-candidate misses so gaps in RAVENSBURGER_SERIES_PREFIXES surface.
        logger.info("No box image for adult-line EAN %s (tried series prefixes %s)", ean, candidates)
    return BarcodeLookupRecord(found=False, box_image_jpeg=None, article_number=None)


@app.post(
    "/api/v1/piece/preview",
    response_model=PiecePreviewResponse,
    dependencies=[
        Depends(rate_limit_by_user(piece_preview_rate_limiter, lambda: settings.RATE_LIMIT_PREVIEW_PER_MINUTE))
    ],
)
async def preview_piece(
    file: Optional[UploadFile] = None,
    include_quality: Annotated[
        bool,
        Query(
            description=(
                "Also run a quick corner-detection pass on the detected region and populate the "
                "lockable/corner_disagreement flags. Those fields are always present in the response "
                "model; when this is omitted/false they are returned as null."
            )
        ),
    ] = False,
) -> PiecePreviewResponse:
    """Detect the puzzle piece region in a live camera frame.

    Stateless and lightweight: the frontend streams downscaled frames here
    while the piece camera is open and overlays the returned outline on the
    preview, so the user can see what will be captured.

    Requires authentication and is rate-limited per-user (see the route's
    ``dependencies``, which composes `get_current_user` with a rate-limit
    check — see `app.rate_limit.rate_limit_by_user`): the client polls this
    in a loop, and an authenticated (or buggy) client could otherwise
    saturate the server's per-frame CV work. Doesn't need the caller's
    identity beyond that, though — it doesn't persist or look up anything.

    Args:
        file: A downscaled camera frame.
        include_quality: When true, populates the best-effort `lockable` and
            `corner_disagreement` piece-geometry flags computed from the
            detected region's polygon (see
            `app.services.piece_geometry.service.quick_quality_from_polygon`).
            Both fields are always present in `PiecePreviewResponse`; when
            omitted/false they are returned as null.

    Returns:
        PiecePreviewResponse with the detected region outline and a confidence
        expressing how piece-like the region is, or found=False.

    Raises:
        HTTPException: If no/invalid file is provided, or the caller has
            exceeded `RATE_LIMIT_PREVIEW_PER_MINUTE` (429).
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    # This endpoint is polled in a loop from the client, so guard against oversized
    # frames. Preview frames are downscaled client-side, so a tight limit is expected.
    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    region = get_piece_detector().detect_region(image)
    if region is None:
        return PiecePreviewResponse(found=False)

    lockable: Optional[bool] = None
    corner_disagreement: Optional[bool] = None
    if include_quality:
        lockable, corner_disagreement = quick_quality_from_polygon(region.polygon)

    x, y, w, h = region.bbox
    return PiecePreviewResponse(
        found=True,
        polygon=[Corner(x=px, y=py) for px, py in region.polygon],
        bbox=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=region.confidence,
        lockable=lockable,
        corner_disagreement=corner_disagreement,
    )


@app.post("/api/v1/puzzle/{puzzle_id}/piece", response_model=PieceResponse)
async def process_piece(
    puzzle_id: str,
    puzzle: Annotated[PuzzleRecord, Depends(get_owned_puzzle)],
    file: Optional[UploadFile] = None,
    remove_bg: Annotated[bool, Query(description="Remove background from piece image using rembg")] = True,
) -> PieceResponse:
    """Process a puzzle piece image.

    Args:
        puzzle_id: ID of the puzzle to match against.
        puzzle: The caller's owned puzzle record; the dependency also 404s
            when the puzzle doesn't exist or belongs to someone else.
        file: The puzzle piece image file.
        remove_bg: Whether to remove background from piece image (default: True).

    Returns:
        PieceResponse: Response containing position, confidence, and optionally cleaned image;
        when the puzzle's grid is known, also the nearest grid cell and its center
        (grid_row/grid_col/snapped_position) for display.

    Raises:
        HTTPException: If puzzle not found or file type is invalid.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if settings.MATCHER == "cnn":
        response = await get_image_processor().process_piece(file, puzzle_id, remove_background=remove_bg)
    else:
        # Pass along the grid estimated at upload time (if any) so the NCC fallback's
        # nominal template size uses the puzzle's real grid instead of the defaults.
        response = await get_classical_matcher().process_piece(
            file, puzzle_id, remove_background=remove_bg, grid_hint=puzzle.grid
        )
    return snap_to_grid(response, puzzle.grid)


def _normalize_points(points: np.ndarray, width: int, height: int) -> List[GeometryPoint]:
    """Normalize Nx2 pixel points to a list of [0, 1] GeometryPoint.

    Args:
        points: Nx2 array of pixel coordinates.
        width: The source image's width in pixels.
        height: The source image's height in pixels.

    Returns:
        The normalized, clamped points.
    """
    return [
        GeometryPoint(x=float(np.clip(x / width, 0.0, 1.0)), y=float(np.clip(y / height, 0.0, 1.0))) for x, y in points
    ]


def _geometry_quality_response(record: PieceGeometryRecord) -> GeometryQuality:
    """Build the GeometryQuality response model from a PieceGeometryRecord.

    Args:
        record: The pipeline's output record.

    Returns:
        The corresponding `GeometryQuality` response model.
    """
    quality = record.quality
    return GeometryQuality(
        is_clean=quality.is_clean,
        corner_disagreement=record.corner_disagreement,
        n_large_components=quality.n_large_components,
        border_touching=quality.border_touching,
        area_ratio=quality.area_ratio,
        solidity=quality.solidity,
    )


def _geometry_record_response(
    record: PieceGeometryRecord, width: int, height: int, include_contour: bool
) -> PieceGeometryRecordResponse:
    """Build the PieceGeometryRecordResponse from a PieceGeometryRecord.

    Args:
        record: The pipeline's output record.
        width: The uploaded photo's width in pixels (for normalization).
        height: The uploaded photo's height in pixels (for normalization).
        include_contour: Whether to include the (large) full contour polyline.

    Returns:
        The corresponding `PieceGeometryRecordResponse` response model.
    """
    corners = _normalize_points(record.corners, width, height) if record.corners is not None else []
    edges = (
        [
            GeometryEdge(
                type=edge.edge_type,  # type: ignore[arg-type]
                dominant_dev=edge.dominant_dev,
                polyline=_normalize_points(edge.polyline, width, height),
            )
            for edge in record.edges
        ]
        if record.edges is not None
        else []
    )
    contour = (
        _normalize_points(record.contour, width, height) if include_contour and record.contour is not None else None
    )
    return PieceGeometryRecordResponse(
        corners=corners,
        corner_confidences=record.corner_confidences or [],
        edges=edges,
        contour=contour,
    )


@app.post("/api/v1/puzzle/{puzzle_id}/piece/geometry", response_model=PieceGeometryUploadResponse)
async def upload_piece_geometry(
    puzzle_id: str,
    puzzle: Annotated[PuzzleRecord, Depends(get_owned_puzzle)],
    file: Optional[UploadFile] = None,
    include_contour: Annotated[
        bool, Query(description="Include the full contour polyline in the response (large)")
    ] = False,
    on_uncertain: Annotated[
        Literal["report", "enroll"],
        Query(
            description=(
                "How to resolve a gray-zone (uncertain) verdict: 'report' returns status=uncertain without "
                "enrolling (default); 'enroll' enrolls the photo as a NEW piece and returns status=new. Use "
                "'enroll' after the client's own confirmation UX — most genuinely-new pieces land in the "
                "gray zone (M7), so a session needs this to enroll beyond its first piece."
            )
        ),
    ] = "report",
) -> PieceGeometryUploadResponse:
    """Extract piece geometry from a photo and dedupe it against this puzzle's piece store.

    Runs the exp28 pipeline (background removal -> contour -> corners ->
    edges -> fingerprint) on the uploaded photo, then compares the resulting
    fingerprint against every piece already enrolled for this puzzle to
    decide whether the photo is a piece seen before, a new piece, or an
    uncertain (gray-zone) match.

    Args:
        puzzle_id: ID of the puzzle this piece photo belongs to.
        puzzle: The caller's owned puzzle record; the dependency also 404s
            when the puzzle doesn't exist or belongs to someone else.
        file: The puzzle piece photo.
        include_contour: Whether to include the full (large) contour polyline.
        on_uncertain: Gray-zone resolution: "report" (default) keeps the
            current conservative behavior; "enroll" enrolls a gray-zone
            photo as a new piece. Matched/new verdicts are unaffected.

    Returns:
        PieceGeometryUploadResponse with the dedupe verdict, quality flags,
        and the extracted geometric record.

    Raises:
        HTTPException: If no/invalid file is provided, the file is too
            large, or the puzzle does not exist.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        probe = Image.open(io.BytesIO(contents))
        width, height = probe.size
        probe.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    record = get_piece_geometry_service().process(contents)
    # A record with a fingerprint always has a measured (non-None) disagreement
    # flag; without a fingerprint the store ignores the flag entirely, so
    # coercing None to False here is safe.
    match = get_piece_geometry_store().add_or_match(
        puzzle_id,
        record.fingerprint,
        record.quality.is_clean,
        bool(record.corner_disagreement),
        enroll_uncertain=on_uncertain == "enroll",
    )

    return PieceGeometryUploadResponse(
        piece_id=match.piece_id,
        status=match.verdict.value,
        match_piece_id=match.match_piece_id,
        z_score=match.z_score,
        lockable=record.lockable,
        quality=_geometry_quality_response(record),
        record=_geometry_record_response(record, width, height, include_contour),
    )


@app.get("/api/v1/puzzle/{puzzle_id}/piece/geometry", response_model=PieceGeometryListResponse)
async def list_piece_geometry(
    puzzle_id: str,
    puzzle: Annotated[PuzzleRecord, Depends(get_owned_puzzle)],
) -> PieceGeometryListResponse:
    """List the pieces enrolled in a puzzle's piece-geometry store.

    Args:
        puzzle_id: ID of the puzzle to list.
        puzzle: The caller's owned puzzle record; the dependency also 404s
            when the puzzle doesn't exist or belongs to someone else.

    Returns:
        PieceGeometryListResponse with one summary (id, edge types, quality
        flags — no polylines) per enrolled piece.

    Raises:
        HTTPException: If the puzzle does not exist.
    """
    enrolled = get_piece_geometry_store().list_pieces(puzzle_id)
    return PieceGeometryListResponse(
        puzzle_id=puzzle_id,
        pieces=[
            PieceGeometrySummary(
                piece_id=piece.piece_id,
                edge_types=list(piece.fingerprint.edge_types),  # type: ignore[arg-type]
                is_clean=piece.is_clean,
                corner_disagreement=piece.corner_disagreement,
            )
            for piece in enrolled
        ],
    )


@app.delete("/api/v1/puzzle/{puzzle_id}/piece/geometry/{piece_id}", status_code=204)
async def delete_piece_geometry(
    puzzle_id: str,
    piece_id: str,
    puzzle: Annotated[PuzzleRecord, Depends(get_owned_puzzle)],
) -> None:
    """Un-enroll one piece from a puzzle's piece-geometry store.

    Mirrors the client deleting a scanned piece from its piece list: the piece
    disappears from the scanner's gallery, and a fresh photo of it reads as a
    new piece again instead of matching the stale enrollment.

    Args:
        puzzle_id: ID of the puzzle the piece belongs to.
        piece_id: ID of the enrolled piece to remove.
        puzzle: The caller's owned puzzle record; the dependency also 404s
            when the puzzle doesn't exist or belongs to someone else.

    Raises:
        HTTPException: If the puzzle does not exist, or it has no piece with
            this id enrolled.
    """
    if not get_piece_geometry_store().remove(puzzle_id, piece_id):
        raise HTTPException(status_code=404, detail="Piece not found")


@app.post("/api/v1/puzzle/{puzzle_id}/generate-piece", response_model=GeneratePieceResponse)
async def generate_piece(
    puzzle_id: str,
    request: GeneratePieceRequest,
    puzzle: Annotated[PuzzleRecord, Depends(get_owned_puzzle)],
) -> GeneratePieceResponse:
    """Generate a realistic jigsaw-shaped piece at the specified position.

    This endpoint creates a puzzle piece with realistic Bezier curve edges
    (tabs and blanks) extracted from the uploaded puzzle image. The piece
    is returned as a PNG with transparent background.

    Args:
        puzzle_id: ID of the puzzle to extract piece from.
        request: Contains center_x, center_y (normalized 0-1 coordinates)
            and optional piece_size_ratio.
        puzzle: The caller's owned puzzle record; the dependency also 404s
            when the puzzle doesn't exist or belongs to someone else.

    Returns:
        GeneratePieceResponse with piece_image (base64 PNG) and piece_config.

    Raises:
        HTTPException: If puzzle not found.
    """
    puzzle_img = Image.open(puzzle.file_path).convert("RGBA")

    # Generate piece with random jigsaw shape
    generator = PieceShapeGenerator(piece_size_ratio=request.piece_size_ratio)
    piece_img, config = generator.generate_piece(
        puzzle_img,
        request.center_x,
        request.center_y,
    )

    # Convert to base64 PNG
    buffer = io.BytesIO()
    piece_img.save(buffer, format="PNG")
    piece_base64 = base64.b64encode(buffer.getvalue()).decode()

    return GeneratePieceResponse(
        piece_image=f"data:image/png;base64,{piece_base64}",
        piece_config=config.to_dict(),
    )


@app.post("/api/v1/puzzle/{puzzle_id}/cut-all", response_model=CutPuzzleResponse)
async def cut_puzzle(
    puzzle_id: str,
    request: CutPuzzleRequest,
    puzzle: Annotated[PuzzleRecord, Depends(get_owned_puzzle)],
) -> CutPuzzleResponse:
    """Cut a puzzle into jigsaw-shaped pieces for manual solving.

    This endpoint cuts the uploaded puzzle image into a grid of jigsaw-shaped
    pieces with realistic Bezier curve edges. Each piece is returned as a PNG
    with transparent background, along with its correct position for game play.

    Requires authentication and ownership of the puzzle (previously this
    endpoint had no auth dependency at all).

    Args:
        puzzle_id: ID of the puzzle to cut.
        request: Contains rows, cols, and optional seed for reproducibility.
        puzzle: The caller's owned puzzle record; the dependency also 404s
            when the puzzle doesn't exist or belongs to someone else.

    Returns:
        CutPuzzleResponse with list of pieces, grid dimensions, and puzzle size.

    Raises:
        HTTPException: If puzzle not found.
    """
    puzzle_img = Image.open(puzzle.file_path)

    # Cut into pieces
    cutter = get_puzzle_cutter()
    pieces, puzzle_width, puzzle_height = cutter.cut_puzzle(
        puzzle_img,
        rows=request.rows,
        cols=request.cols,
        seed=request.seed,
    )

    return CutPuzzleResponse(
        pieces=pieces,
        grid={"rows": request.rows, "cols": request.cols},
        puzzle_width=puzzle_width,
        puzzle_height=puzzle_height,
    )
