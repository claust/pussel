"""Main FastAPI application module for the puzzle solver."""

import os
import uuid
from typing import Dict, Optional

from app.config import settings
from app.models.puzzle_model import PieceResponse, PuzzleResponse
from app.services.image_processor import ImageProcessor
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/api/v1/puzzle/upload", response_model=PuzzleResponse)
async def upload_puzzle(file: Optional[UploadFile] = None) -> PuzzleResponse:
    """Upload a complete puzzle image.

    Args:
        file: The puzzle image file.

    Returns:
        PuzzleResponse: Response containing the puzzle ID.

    Raises:
        HTTPException: If file size exceeds limit or file type is invalid.
    """
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
    file: Optional[UploadFile] = None,
) -> PieceResponse:
    """Process a puzzle piece image.

    Args:
        puzzle_id: ID of the puzzle to match against.
        file: The puzzle piece image file.

    Returns:
        PieceResponse: Response containing position and confidence.

    Raises:
        HTTPException: If puzzle not found or file type is invalid.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if puzzle_id not in puzzle_images:
        raise HTTPException(status_code=404, detail="Puzzle not found")

    processor = ImageProcessor()
    return await processor.process_piece(file)
