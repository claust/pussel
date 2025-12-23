import 'dart:typed_data';

import 'package:flutter/material.dart';

/// A dialog for selecting piece rotation.
class RotationSelectorDialog extends StatefulWidget {
  const RotationSelectorDialog({required this.piecePreview, super.key});

  /// Preview image bytes of the selected piece.
  final Uint8List piecePreview;

  @override
  State<RotationSelectorDialog> createState() => _RotationSelectorDialogState();
}

class _RotationSelectorDialogState extends State<RotationSelectorDialog> {
  int _selectedRotation = 0;

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: const Text('Select Rotation'),
    content: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Piece preview with rotation applied
        Container(
          width: 150,
          height: 150,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey),
            borderRadius: BorderRadius.circular(8),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Transform.rotate(
              angle: _selectedRotation * 3.14159 / 180,
              child: Image.memory(widget.piecePreview, fit: BoxFit.cover),
            ),
          ),
        ),
        const SizedBox(height: 16),
        // Rotation buttons
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _buildRotationButton(0, '0\u00B0'),
            _buildRotationButton(90, '90\u00B0'),
            _buildRotationButton(180, '180\u00B0'),
            _buildRotationButton(270, '270\u00B0'),
          ],
        ),
      ],
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.pop(context),
        child: const Text('Cancel'),
      ),
      ElevatedButton(
        onPressed: () => Navigator.pop(context, _selectedRotation),
        child: const Text('Confirm'),
      ),
    ],
  );

  Widget _buildRotationButton(int rotation, String label) => InkWell(
    onTap: () => setState(() => _selectedRotation = rotation),
    borderRadius: BorderRadius.circular(8),
    child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color:
            _selectedRotation == rotation
                ? Theme.of(context).colorScheme.primary
                : Colors.grey.shade200,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: _selectedRotation == rotation ? Colors.white : Colors.black,
          fontWeight: FontWeight.bold,
        ),
      ),
    ),
  );
}
