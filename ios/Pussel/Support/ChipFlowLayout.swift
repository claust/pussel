import SwiftUI

/// Lays capsule chips out left to right, wrapping to a new row whenever the
/// next chip would overflow the proposed width, with every row centred.
///
/// SwiftUI has no built-in flow layout, and the alternatives don't fit the
/// piece-count picker: an `HStack` clips its overflow and a horizontal
/// `ScrollView` hides chips past the edge — including "Other", the only route
/// to a count the presets don't cover.
struct ChipFlowLayout: Layout {
  var spacing: CGFloat = 8

  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
    let proposed = proposal.width ?? .infinity
    let rows = rows(for: subviews, width: proposed)
    // A chip too wide to wrap (long localization, large Dynamic Type) sits on
    // a row of its own that overflows; report at most what we were offered so
    // the parent isn't asked for width it won't hand back.
    let width = min(rows.map(\.width).max() ?? 0, proposed)
    let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(rows.count - 1, 0))
    return CGSize(width: width, height: height)
  }

  func placeSubviews(
    in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void
  ) {
    var y = bounds.minY
    for row in rows(for: subviews, width: bounds.width) {
      // max() so an overflowing row starts at the leading edge rather than
      // being centred into a negative origin, which would clip both ends.
      var x = bounds.minX + max((bounds.width - row.width) / 2, 0)
      for index in row.indices {
        let size = subviews[index].sizeThatFits(.unspecified)
        subviews[index].place(
          at: CGPoint(x: x, y: y + (row.height - size.height) / 2),
          proposal: ProposedViewSize(size)
        )
        x += size.width + spacing
      }
      y += row.height + spacing
    }
  }

  private struct Row {
    var indices: [Subviews.Index] = []
    var width: CGFloat = 0
    var height: CGFloat = 0
  }

  private func rows(for subviews: Subviews, width: CGFloat) -> [Row] {
    var rows: [Row] = []
    var row = Row()
    for index in subviews.indices {
      let size = subviews[index].sizeThatFits(.unspecified)
      let widthWithChip = row.indices.isEmpty ? size.width : row.width + spacing + size.width
      // Keep a lone oversized chip on its own row rather than dropping it.
      if widthWithChip > width, !row.indices.isEmpty {
        rows.append(row)
        row = Row()
      }
      row.width = row.indices.isEmpty ? size.width : row.width + spacing + size.width
      row.height = max(row.height, size.height)
      row.indices.append(index)
    }
    if !row.indices.isEmpty { rows.append(row) }
    return rows
  }
}
