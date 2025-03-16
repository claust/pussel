import 'package:flutter/material.dart';
import 'ui/screens/home_screen.dart';
import 'ui/theme/app_theme.dart';

// Main entry point to the app
void main() {
  runApp(const PusselApp());
}

class PusselApp extends StatelessWidget {
  const PusselApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Pussel',
    theme: AppTheme.lightTheme,
    darkTheme: AppTheme.darkTheme,
    home: const HomeScreen(),
    debugShowCheckedModeBanner: false,
  );
}
