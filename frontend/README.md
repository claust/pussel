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
│   ├── config/           # App configuration
│   ├── models/           # Data models
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

- Flutter SDK (version 3.7.0 or higher)
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
flutter run
```

### Backend Configuration

The app is configured to connect to a backend running at:
- Android Emulator: `http://10.0.2.2:8000`
- iOS Simulator: `http://localhost:8000`

To change the backend URL, modify the `baseUrl` in `lib/config/api_config.dart`.

## Testing on Android Emulator

1. Start the Android emulator
2. Ensure the backend API is running
3. Run the app with:

```bash
flutter run
```

## Development

### Code Style

This project follows the official [Flutter style guide](https://flutter.dev/docs/development/tools/formatting) and uses the Flutter formatter:

```bash
flutter format .
```

### Adding Dependencies

To add new dependencies, modify the `pubspec.yaml` file and run:

```bash
flutter pub get
```

### Building for Release

To build a release version of the app:

```bash
# For Android
flutter build apk --release

# For iOS
flutter build ios --release
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
