# Puzzle Shape Generation Improvement Suggestions

Based on an analysis of the **Reference (Blue)** vs. **Generated (Red)** pieces in `reference_comparison.png` and the provided Bézier structure.

## A) Curve Evaluation: Split vs. Sufficient

The current **5-curve model is insufficient** for capturing the variety of real puzzle shapes.

*   **Split the Bulb Cap (C3):** Reference pieces 2, 3, and 5 show "flat-topped" or "rounded-rectangle" bulbs. A single cubic Bézier (C3) naturally tends toward a circular or parabolic arc. Splitting C3 into **three segments** (Corner Arc → Flat/Slightly Curved Top → Corner Arc) will allow the "squarish" look seen in the reference.
*   **Split the Shoulders (C1 & C5):** The reference pieces have very broad, organic "waists" where the piece body transitions into the neck. Splitting C1/C5 into a straight segment and a dedicated "Shoulder Transition" segment would allow for better control of the flare width.

## B) Handle Shifting (P1, P2) for Organic Look

To match the "rounded-rectangle" and "broad-shouldered" reference shapes, adjust the handles as follows:

*   **For the Bulb (C3.P1 & C3.P2):** Move these handles horizontally **outward** (away from each other) and slightly **upward**. This pushes the curvature into the corners, flattening the top of the bulb.
*   **For the Neck (C2.P1 & C2.P2):** The reference pieces have much "thicker" necks at the base. Move **C2.P1** (the lower neck handle) horizontally **away from the center line**. This creates the "flared base" seen in Reference 4 and 6.
*   **For the Shoulders (C1.P2):** In the reference, the transition from the straight edge to the neck is very gradual. Move the exit handle of the shoulder (**C1.P2**) further **away from the neck** to lengthen the transition zone.

## C) Suggested Parameters and Geometric Instructions

Add or adjust the following parameters to capture the "shoulders" and "waists":

1.  **Bulb Squareness (Top Flatness):**
    *   *Logic:* Increase the magnitude of handles `C3.P1` and `C3.P2` while keeping them horizontal.
    *   *Instruction:* **"Extend the horizontal distance of the Bulb Cap handles (C3.P1, C3.P2) toward the bulb's outer bounds to flatten the apex."**

2.  **Shoulder Flare (Base Width):**
    *   *Logic:* Current generated pieces (Overlap 6) have necks that "pinch" too sharply at the base.
    *   *Instruction:* **"Move the handles of the Neck Ascent (C2.P1) and Neck Descent (C4.P2) further apart horizontally to create a more flared, organic base transition."**

3.  **Waist Curvature (Neck Fullness):**
    *   *Logic:* The reference pieces often have a "straighter" neck ascent before curving into the bulb.
    *   *Instruction:* **"Align the vertical positions of C2.P1 and C2.P2 more closely to create a steeper, less 'pinched' neck column before it opens into the bulb."**

4.  **Tangential Continuity (C1 to C2):**
    *   *Instruction:* **"Ensure the handle C1.P2 and C2.P1 are collinear with the junction point C2.P0 to prevent the sharp 'kink' visible in the generated piece shoulders."** (Visible in Overlap 1 and 6).
