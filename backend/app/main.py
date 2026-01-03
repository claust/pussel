"""Main FastAPI application module for the puzzle solver."""

import base64
import io
import os
import uuid
from typing import Annotated, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.auth.dependencies import get_current_user
from app.auth.service import AuthService, get_auth_service
from app.config import settings
from app.models.puzzle_model import (
    CutPuzzleRequest,
    CutPuzzleResponse,
    GeneratePieceRequest,
    GeneratePieceResponse,
    PieceResponse,
    PuzzleResponse,
)
from app.models.user_model import GoogleAuthRequest, TokenResponse, User
from app.services.image_processor import get_image_processor
from app.services.piece_shape import PieceShapeGenerator
from app.services.puzzle_cutter import get_puzzle_cutter

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

# Store puzzle images in memory for demo
puzzle_images: Dict[str, str] = {}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# =============================================================================
# Authentication Endpoints
# =============================================================================


@app.post("/api/v1/auth/google", response_model=TokenResponse)
async def google_auth(
    request: GoogleAuthRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Authenticate with Google OAuth.

    Exchanges a Google ID token for an application JWT token.

    Args:
        request: The Google authentication request containing the ID token.
        auth_service: The authentication service.

    Returns:
        TokenResponse containing the access token and user information.

    Raises:
        HTTPException: If the Google token is invalid.
    """
    user_info = auth_service.verify_google_token(request.id_token)
    if user_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid Google token",
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
) -> PuzzleResponse:
    """Upload a complete puzzle image.

    Args:
        current_user: The authenticated user.
        file: The puzzle image file.

    Returns:
        PuzzleResponse: Response containing the puzzle ID.

    Raises:
        HTTPException: If file size exceeds limit or file type is invalid.
    """
    # Note: current_user can be used to associate puzzles with users in the future
    _ = current_user  # Acknowledge the parameter is intentionally unused for now
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    puzzle_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{puzzle_id}.jpg")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    puzzle_images[puzzle_id] = file_path
    return PuzzleResponse(puzzle_id=puzzle_id)


@app.post("/api/v1/puzzle/{puzzle_id}/piece", response_model=PieceResponse)
async def process_piece(
    puzzle_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    file: Optional[UploadFile] = None,
    remove_bg: Annotated[bool, Query(description="Remove background from piece image using rembg")] = True,
) -> PieceResponse:
    """Process a puzzle piece image.

    Args:
        puzzle_id: ID of the puzzle to match against.
        current_user: The authenticated user.
        file: The puzzle piece image file.
        remove_bg: Whether to remove background from piece image (default: True).

    Returns:
        PieceResponse: Response containing position, confidence, and optionally cleaned image.

    Raises:
        HTTPException: If puzzle not found or file type is invalid.
    """
    _ = current_user  # Acknowledge the parameter is intentionally unused for now
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if puzzle_id not in puzzle_images:
        raise HTTPException(status_code=404, detail="Puzzle not found")

    processor = get_image_processor()
    return await processor.process_piece(file, puzzle_id, remove_background=remove_bg)


@app.post("/api/v1/puzzle/{puzzle_id}/generate-piece", response_model=GeneratePieceResponse)
async def generate_piece(
    puzzle_id: str,
    request: GeneratePieceRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> GeneratePieceResponse:
    """Generate a realistic jigsaw-shaped piece at the specified position.

    This endpoint creates a puzzle piece with realistic Bezier curve edges
    (tabs and blanks) extracted from the uploaded puzzle image. The piece
    is returned as a PNG with transparent background.

    Args:
        puzzle_id: ID of the puzzle to extract piece from.
        request: Contains center_x, center_y (normalized 0-1 coordinates)
            and optional piece_size_ratio.
        current_user: The authenticated user.

    Returns:
        GeneratePieceResponse with piece_image (base64 PNG) and piece_config.

    Raises:
        HTTPException: If puzzle not found.
    """
    _ = current_user  # Acknowledge the parameter is intentionally unused for now
    if puzzle_id not in puzzle_images:
        raise HTTPException(status_code=404, detail="Puzzle not found")

    # Load puzzle image
    puzzle_path = puzzle_images[puzzle_id]
    puzzle_img = Image.open(puzzle_path).convert("RGBA")

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
) -> CutPuzzleResponse:
    """Cut a puzzle into jigsaw-shaped pieces for manual solving.

    This endpoint cuts the uploaded puzzle image into a grid of jigsaw-shaped
    pieces with realistic Bezier curve edges. Each piece is returned as a PNG
    with transparent background, along with its correct position for game play.

    Args:
        puzzle_id: ID of the puzzle to cut.
        request: Contains rows, cols, and optional seed for reproducibility.

    Returns:
        CutPuzzleResponse with list of pieces, grid dimensions, and puzzle size.

    Raises:
        HTTPException: If puzzle not found.
    """
    if puzzle_id not in puzzle_images:
        raise HTTPException(status_code=404, detail="Puzzle not found")

    # Load puzzle image
    puzzle_path = puzzle_images[puzzle_id]
    puzzle_img = Image.open(puzzle_path)

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
