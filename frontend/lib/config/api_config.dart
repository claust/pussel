class ApiConfig {
  static const String baseUrl =
      'http://10.0.2.2:8000'; // Use this for Android Emulator
  // static const String baseUrl = 'http://localhost:8000'; // Use this for iOS Simulator

  // API endpoints
  static const String healthEndpoint = '/health';
  static const String uploadPuzzleEndpoint = '/api/v1/puzzle/upload';
  static String processPieceEndpoint(String puzzleId) =>
      '/api/v1/puzzle/$puzzleId/piece';

  // Timeout durations
  static const int connectionTimeout = 30000; // 30 seconds
  static const int receiveTimeout = 30000; // 30 seconds

  // Upload constraints
  static const int maxUploadSizeMb = 10; // 10 MB
}
