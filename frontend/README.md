# Pussel Frontend

A Flutter mobile app for the Pussel puzzle solver project. This app allows users to take photos of jigsaw puzzles and individual pieces, and uses computer vision to help solve the puzzle.

## Features

- Take photos of complete puzzles
- Take photos of individual puzzle pieces
- Upload images to the backend for processing
- View puzzle pieces with their positions and orientations
- Modern Material Design 3 UI with light and dark theme support

## Project Structure

```
frontend/
├── lib/
│   ├── config/           # API configuration
│   ├── models/           # Data models (Puzzle, Piece)
│   ├── services/         # API and camera services
│   ├── ui/               # UI components
│   │   ├── screens/      # Full screens
│   │   ├── widgets/      # Reusable widgets
│   │   └── theme/        # App theming
│   └── utils/            # Utility functions
├── assets/               # Static assets
└── test/                 # Tests
```

## Getting Started

### Prerequisites

- Flutter SDK (version 3.29.2 or higher)
- Dart SDK (version 3.7.2 or higher)
- Android Studio or Xcode for running on emulators/simulators
- A running backend API service (see backend README)

### Installation

1. Ensure Flutter is installed and available in your PATH
2. Clone the repository and navigate to the frontend directory
3. Install dependencies:

```bash
cd frontend
flutter pub get
```

4. Run the app:

```bash
# For mobile/desktop devices
flutter run

# For web (Chrome)
flutter run -d chrome
```

Note: When deploying to web, specify Chrome explicitly with `-d chrome` rather than using `-d web`, as the latter may not be recognized as a valid device ID.

### Backend Configuration

The app is configured to connect to a backend running at the URL specified in `lib/config/api_config.dart`. The default settings are:
- Base URL: Defined in ApiConfig.baseUrl
- Timeout settings: Configurable via ApiConfig constants
- Endpoints: Defined in ApiConfig class

## Development

### Code Style and Linting

This project uses strict linting rules to ensure high code quality:

- Custom rules defined in `analysis_options.yaml`
- Enforced trailing commas for better git diffs
- Sorted imports and constructors
- Preferred use of expression function bodies
- Required final variables where appropriate

To check your code against these rules:

```bash
flutter analyze
```

To automatically format your code:

```bash
dart format .
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. After cloning the repository, install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hooks will:
- Run dart format
- Check dart analysis
- Apply automated fixes
- Ensure files end with newlines
- Prevent trailing whitespace

### Dependencies

Key dependencies used in this project:

- **API and networking**: dio, http
- **State management**: provider
- **Image and camera**: camera, image_picker, path_provider

To add new dependencies, modify the `pubspec.yaml` file and run:

```bash
flutter pub get
```

### Continuous Integration

This project uses GitHub Actions for CI/CD, which:
- Runs linting and tests on every PR
- Builds APK and web versions
- Uploads build artifacts

## Building for Release

To build a release version of the app:

```bash
# For Android
flutter build apk --release

# For iOS
flutter build ios --release

# For web
flutter build web --release
```

## Testing

Run the tests with:

```bash
flutter test
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
