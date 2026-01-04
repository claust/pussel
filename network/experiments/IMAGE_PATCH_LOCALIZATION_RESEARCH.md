# Image Patch Localization Research

Research on approaches to locate a small part of a picture inside another picture.
Compiled January 2026 to inform future experiments after exp20-21 rotation failures.

---

## Context: The Puzzle Piece Problem

In exp20-21, we achieved:
- **Position prediction: 73%** - texture correlation works well
- **Rotation prediction: 25%** - complete failure (random baseline)

The rotation correlation approach that worked for square pieces (93-95% accuracy) completely fails for realistic Bezier-curve pieces. This research explores alternative approaches.

---

## 1. Template Matching (Direct Pixel Comparison)

### Normalized Cross-Correlation (NCC)

- Slides template across image, computes correlation at each position
- OpenCV: `cv.matchTemplate()` with `CV_TM_CCOEFF_NORMED`
- Fast with FFT, but **not rotation-invariant** by default
- For rotation: must search all rotation angles (slow) or use Sub-NCC algorithm for 95%+ accuracy

**Optimizations:**
- **Image pyramids** - coarse-to-fine search reduces computation
- **Fastest Image Pattern Matching** - 4-128x speedup over standard NCC using modified rotation matrix

**Limitations:**
- Standard template matching is NOT rotation/scale invariant
- Must explicitly search rotation space, which is expensive

**References:**
- [OpenCV Template Matching Tutorial](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [Sub-NCC Algorithm Paper](https://www.aimspress.com/article/doi/10.3934/mbe.2022442)
- [Fastest Image Pattern Matching (GitHub)](https://github.com/DennisLiu1993/Fastest_Image_Pattern_Matching)

---

## 2. Feature-Based Matching (Keypoint Detection)

### Comparison Table

| Method | Speed | Rotation Invariant | Scale Invariant | Best For |
|--------|-------|-------------------|-----------------|----------|
| **SIFT** | Slow | Yes | Yes | Accuracy-critical |
| **SURF** | Medium | Yes | Yes | Balanced |
| **ORB** | Very Fast | Yes | No | Real-time |

### How It Works

1. Detect keypoints (corners, blobs) in both images
2. Compute descriptors (128-dim for SIFT, binary for ORB)
3. Match descriptors using nearest-neighbor
4. Filter with RANSAC for geometric consistency

### SIFT (Scale-Invariant Feature Transform)

- Created in 2004 by D. Lowe
- Uses Difference of Gaussian (DoG) for keypoint detection
- 128-dimensional descriptor
- **Most accurate and robust** across transformations
- Slower than alternatives

### ORB (Oriented FAST and Rotated BRIEF)

- Introduced by Rublee et al. in 2011
- Combines FAST keypoint detector with BRIEF descriptor
- Adds rotational invariance to BRIEF
- **Two orders of magnitude faster than SIFT**
- Does NOT have scale invariance
- More sensitive to noise

### Key Insight

**SIFT/ORB keypoints are inherently rotation-invariant by design.** This could be valuable for the puzzle rotation problem since the current correlation approach fails.

**References:**
- [SIFT vs ORB Comparison (Medium)](https://medium.com/@beauc_37732/comparing-sift-and-orb-for-feature-matching-a-visual-and-practical-exploration-6c194c72e4d6)
- [ORB Paper (IEEE)](https://ieeexplore.ieee.org/document/6126544/)
- [Local Feature Matching Survey](https://arxiv.org/html/2401.17592v2)

---

## 3. Deep Learning Feature Matching

### SuperGlue (CVPR 2020)

- Graph neural network for matching keypoints
- Uses attention for context aggregation
- Solves optimal transport for assignments
- 12M parameters, real-time on GPU
- L=9 layers of alternating self- and cross-attention with 4 heads

**Architecture:**
1. Keypoint encoder maps positions + visual descriptors to single vector
2. Alternating self- and cross-attention layers
3. Optimal matching layer using Sinkhorn algorithm

### LoFTR (CVPR 2021)

- **Detector-free** - no keypoint detection step
- Transformer with self/cross attention
- Produces dense matches even in **low-texture areas**
- State-of-the-art on indoor/outdoor benchmarks

**Architecture:**
1. Local feature CNN extracts coarse and fine feature maps
2. Flatten to 1-D vectors with positional encoding
3. Self-attention and cross-attention layers
4. Coarse-to-fine matching

**Key Advantage:** Works where traditional methods fail (textureless regions, repetitive patterns)

### Efficient LoFTR

- Semi-dense matching with sparse-like speed
- Practical for real-time applications

**References:**
- [SuperGlue GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [SuperGlue Paper](https://arxiv.org/abs/1911.11763)
- [LoFTR Project Page](https://zju3dv.github.io/loftr/)
- [LoFTR GitHub](https://github.com/zju3dv/LoFTR)

---

## 4. Edge/Shape-Based Matching

### Chamfer Matching

- Computes distance transform of edge image
- Measures how well template edges align with image edges
- Very robust to segmentation quality
- **Linear complexity** in contour length (not area)

**How it works:**
1. Extract edges from both images
2. Compute distance transform of target image edges
3. Sum distances from template edge points to nearest target edge
4. Lower score = better match

**Advantages:**
- Robust to poor segmentation
- Can be fully automated
- Works with "poor quality" images

### Hausdorff Distance

- Measures max distance from any model point to nearest image point
- h(A,B) = max over all points in A of (min distance to any point in B)
- **Tolerant of position errors** from edge detectors
- No point correspondence needed

**Key Property:** If h(A,B) = e, then ALL points in A are within distance e of B.

### Oriented Chamfer Matching

- Matches edge **orientation** AND location
- Better discrimination in cluttered scenes
- Significantly improves detection performance
- Speeds up branch-and-bound search

**References:**
- [Chamfer Matching Overview (ScienceDirect)](https://www.sciencedirect.com/topics/engineering/chamfer-matching)
- [Hausdorff Distance Lecture (Cornell)](https://www.cs.cornell.edu/courses/cs664/2003fa/handouts/664-l3-matching-03.pdf)
- [Chamfer System (Gavrila)](http://www.gavrila.net/Research/Chamfer_System/chamfer_system.html)

---

## 5. Puzzle-Specific Approaches

### From Academic Literature

**Boundary-Centered Polar Encoding**
- Rotation-invariant representation of piece boundaries
- Uses relative directions (changes between line segments) instead of absolute
- Encoding is independent of piece placement on xy plane

**Corner Detection with Rotation Invariance**
- Extract characteristic points (high curvature) from piece edges
- Use rotationally invariant corner detection algorithm
- Compare geometrical and color features at these points

**Pairwise Compatibility Measures**
- Combine Gist descriptors with color distance
- Gradient + color features improve assembly
- Rotation-based strategies for working on multiple parts

**Vision Transformers for Puzzles**
- Recent work uses transformers to determine piece placement
- Encoder uses information at piece edges
- **Limitation:** Current methods assume known orientation
- For unknown orientation: need rotation-invariant embeddings

**References:**
- [Vision-Based Jigsaw Puzzle Solving (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10422444/)
- [Jigsolved (Medium)](https://medium.com/cornell-tech/jigsolved-computer-vision-to-solve-jigsaw-puzzles-70b8ad8099e5)
- [Solving Jigsaw Puzzles with Vision Transformers](https://link.springer.com/article/10.1007/s10044-025-01484-z)

---

## 6. Summary: Applicability to Puzzle Problem

### Current Approach Analysis

| Approach | For Position | For Rotation |
|----------|--------------|--------------|
| **Current (texture correlation)** | Works (73%) | Fails (25%) |
| **Masking (exp21)** | No change | No improvement |

### Promising Alternatives for Rotation

| Approach | Rationale | Complexity |
|----------|-----------|------------|
| **SIFT/ORB keypoints** | Inherently rotation-invariant by design | Medium |
| **Chamfer matching on edges** | Match Bezier contour shape directly | Medium |
| **Hausdorff on silhouettes** | Compare piece outlines | Low |
| **LoFTR** | Dense matching, handles low-texture | High |

### Recommended Experiments

1. **SIFT/ORB for rotation detection**
   - Extract keypoints from piece
   - Match against puzzle at each candidate rotation
   - Select rotation with most/best keypoint matches
   - Keypoints are rotation-invariant, so this should generalize

2. **Chamfer matching on piece edges**
   - Extract Bezier edge contour from piece
   - Compare against puzzle region edges using chamfer distance
   - Edge shapes are consistent across pieces (tabs/blanks)

3. **Separate position and rotation pipelines**
   - Use current texture correlation for position (works)
   - Use edge-based method for rotation (new)
   - Two-stage inference

4. **Data augmentation approach**
   - Train with all 4 rotations of each piece as separate samples
   - Force model to learn rotation-invariant texture features
   - Simpler than architectural changes

### Key Insight

The rotation problem may be fundamentally different from the position problem:
- **Position**: "Where does this texture belong?" - texture correlation works
- **Rotation**: "Which way is up?" - may require edge/shape analysis, not texture

The irregular Bezier edges are actually **information** we're not using. Traditional puzzle solvers match pieces by their edge shapes. We should consider edge-based approaches for rotation while keeping texture-based for position.

---

## Next Steps

- [ ] Experiment with SIFT keypoint matching for rotation
- [ ] Implement chamfer distance on piece silhouettes
- [ ] Try data augmentation with all rotations
- [ ] Consider two-stage architecture (texture for position, edges for rotation)
