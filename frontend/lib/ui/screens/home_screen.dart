import 'package:flutter/foundation.dart';
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
              // Web platform notice
              if (kIsWeb)
                Container(
                  padding: const EdgeInsets.all(12),
                  margin: const EdgeInsets.only(bottom: 24),
                  decoration: BoxDecoration(
                    color: Colors.amber.shade100,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.amber.shade300),
                  ),
                  child: const Column(
                    children: [
                      Icon(Icons.info_outline, color: Colors.amber),
                      SizedBox(height: 8),
                      Text(
                        'Camera functionality is limited in web browsers. For the best experience, please use a mobile device.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Colors.brown),
                      ),
                    ],
                  ),
                ),

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
