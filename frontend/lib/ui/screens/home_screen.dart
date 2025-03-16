import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import 'camera_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Pussel'), centerTitle: true),
    body: SafeArea(
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo or icon
              const Icon(
                Icons.extension,
                size: 80,
                color: AppTheme.primaryColor,
              ),
              const SizedBox(height: 24),

              // Title
              Text(
                'Puzzle Solver',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 8),

              // Subtitle
              Text(
                'Solve jigsaw puzzles with computer vision',
                style: Theme.of(context).textTheme.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),

              // New Puzzle Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder:
                            (context) =>
                                const CameraScreen(mode: CameraMode.puzzle),
                      ),
                    );
                  },
                  child: const Padding(
                    padding: EdgeInsets.all(12.0),
                    child: Text('New Puzzle', style: TextStyle(fontSize: 16)),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // About Button
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: () {
                    showAboutDialog(
                      context: context,
                      applicationName: 'Pussel',
                      applicationVersion: '1.0.0',
                      applicationLegalese: '© 2024',
                      children: [
                        const SizedBox(height: 24),
                        const Text(
                          'A computer vision-based puzzle solver application '
                          'that helps users solve jigsaw puzzles.',
                        ),
                      ],
                    );
                  },
                  child: const Padding(
                    padding: EdgeInsets.all(12.0),
                    child: Text('About', style: TextStyle(fontSize: 16)),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}
