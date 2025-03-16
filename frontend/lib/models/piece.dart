class Position {
  const Position({required this.x, required this.y});

  factory Position.fromJson(Map<String, dynamic> json) => Position(
    x: (json['x'] as num).toDouble(),
    y: (json['y'] as num).toDouble(),
  );
  final double x;
  final double y;

  Map<String, dynamic> toJson() => {'x': x, 'y': y};
}

class Piece {
  const Piece({
    required this.position,
    required this.confidence,
    required this.rotation,
    this.localImagePath,
  });

  factory Piece.fromJson(Map<String, dynamic> json) => Piece(
    position: Position.fromJson(json['position'] as Map<String, dynamic>),
    confidence: (json['confidence'] as num).toDouble(),
    rotation: json['rotation'] as int,
  );
  final Position position;
  final double confidence;
  final int rotation;
  final String? localImagePath;

  Map<String, dynamic> toJson() => {
    'position': position.toJson(),
    'confidence': confidence,
    'rotation': rotation,
  };
}
