# Puzzle Solver Backend

A FastAPI-based backend service for the Puzzle Solver application that helps users solve jigsaw puzzles using computer vision.

## Features

- Upload complete puzzle images
- Process individual puzzle pieces
- Mock implementation of puzzle piece detection (to be replaced with ML model)
- REST API endpoints
- File upload handling
- Comprehensive test suite

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file (optional):
```bash
cp .env.example .env
```

## Running the Application

Start the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## API Endpoints

### Upload Complete Puzzle
```
POST /api/v1/puzzle/upload
```
- Accepts multipart form data with image file
- Returns puzzle ID

### Process Puzzle Piece
```
POST /api/v1/puzzle/{puzzle_id}/piece
```
- Accepts multipart form data with piece image
- Returns predicted position and confidence score

### Health Check
```
GET /health
```
- Returns API health status

## Running Tests

```bash
pytest
```

## Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── models/          
│   │   └── puzzle_model.py
│   └── services/
│       └── image_processor.py
├── tests/
│   └── test_main.py
├── requirements.txt
└── README.md
```

## Future Improvements

1. Implement actual computer vision model for puzzle piece detection
2. Add authentication
3. Add rate limiting
4. Implement proper image validation
5. Add image compression
6. Implement proper error handling and logging
7. Add database for storing puzzle and piece information 