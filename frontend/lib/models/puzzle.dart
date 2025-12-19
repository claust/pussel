class Puzzle {
  const Puzzle({required this.puzzleId, this.imageUrl, this.localImagePath});

  factory Puzzle.fromJson(Map<String, dynamic> json) => Puzzle(
    puzzleId: json['puzzle_id'] as String,
    imageUrl: json['image_url'] as String?,
  );
  final String puzzleId;
  final String? imageUrl;
  final String? localImagePath;

  Map<String, dynamic> toJson() => {
    'puzzle_id': puzzleId,
    'image_url': imageUrl,
  };
}
