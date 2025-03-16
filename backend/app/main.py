import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models.puzzle_model import PieceResponse, PuzzleResponse
from app.services.image_processor import ImageProcessor

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Default file argument
DEFAULT_FILE = File(...)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Check if the service is healthy."""
    return {"status": "healthy"}


@app.post("/api/v1/puzzle/upload", response_model=PuzzleResponse)
async def upload_puzzle(file: UploadFile = DEFAULT_FILE) -> PuzzleResponse:
    """Upload a puzzle image for processing.

    Args:
        file: The puzzle image file to upload.

    Returns:
        PuzzleResponse: The response containing the puzzle ID.

    Raises:
        HTTPException: If the file is not an image or if there's an error saving it.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique ID for the puzzle
    puzzle_id = str(uuid.uuid4())

    try:
        # Save the file
        file_path = UPLOAD_DIR / f"{puzzle_id}.jpg"
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PuzzleResponse(puzzle_id=puzzle_id)


@app.post("/api/v1/puzzle/{puzzle_id}/piece", response_model=PieceResponse)
async def process_piece(
    puzzle_id: str, file: UploadFile = DEFAULT_FILE
) -> PieceResponse:
    """Process a puzzle piece and find its position in the puzzle.

    Args:
        puzzle_id: The ID of the puzzle to match against.
        file: The puzzle piece image file.

    Returns:
        PieceResponse: The response containing the piece's position and confidence.

    Raises:
        HTTPException: If the puzzle doesn't exist, the file is not an image,
                      or if there's an error processing the piece.
    """
    # Check if puzzle exists
    puzzle_path = UPLOAD_DIR / f"{puzzle_id}.jpg"
    if not puzzle_path.exists():
        raise HTTPException(status_code=404, detail="Puzzle not found")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Process the piece
        piece_data = await file.read()
        processor = ImageProcessor()
        result = processor.process_piece(piece_data, str(puzzle_path))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
