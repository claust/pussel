# Pussel - Puzzle Solver

A computer vision-based puzzle solver application that helps users solve jigsaw puzzles. The application consists of a FastAPI backend for image processing and a Flutter frontend for mobile interaction.

## Project Structure

```
pussel/
├── backend/           # Python FastAPI backend
└── frontend/         # Flutter mobile app (coming soon)
```

## Features

- FastAPI backend with:
  - Puzzle image upload and processing
  - ⚠️ **Note**: Currently uses mock implementation (see [Model Architecture Analysis](backend/MODEL_ARCHITECTURE_ANALYSIS.md))
  - Comprehensive test suite with high coverage
  - Strong type checking and linting
  - Continuous Integration with GitHub Actions
  - Code coverage reporting with Codecov
- Flutter mobile app (coming soon)

## Getting Started

See the individual README files in each directory for setup and running instructions:

- [Backend README](backend/README.md)
- Frontend README (coming soon)

### Machine Learning Model Status

**⚠️ Important**: The current implementation uses a mock system that returns random values, not a trained ML model.

For details on model architecture and implementation roadmap:
- [Model Architecture Analysis](backend/MODEL_ARCHITECTURE_ANALYSIS.md) - Comprehensive technical analysis
- [Quick Start Guide](backend/QUICK_START_GUIDE.md) - Immediate next steps for implementation

## Development

The project follows best practices for code quality and testing:
- Pre-commit hooks for automated checks
- Comprehensive test suite
- Continuous Integration with GitHub Actions
- Code coverage tracking with Codecov
- Type checking and linting
- Automated code formatting

## License

This project is licensed under the MIT License - see the LICENSE file for details.
