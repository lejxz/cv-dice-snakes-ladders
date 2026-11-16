# 2. Related Literature

**Project Title:** Self-Verifying Real-Time Dice Recognition Using Blob Detection and Morphology for a Computer Vision-Driven Snakes and Ladders Game

---

## Overview of Literature Domains

The proposed system bridges four established research areas in computer vision and human-computer interaction (HCI):
1. **Classical Image Segmentation, Mathematical Morphology, and Blob Analysis** — foundational pixel- and region-level operators used for die localization and pip extraction without deep learning.
2. **Automated Dice and Pip Recognition Systems** — prior machine vision methods for segmenting dice, counting pips, and exploiting structural pip geometry.
3. **Selective Classification (Reject Option) and Temporal Consensus** — theoretical and practical frameworks for refusing uncertain predictions and aggregating multi-frame evidence in video streams.
4. **Tangible & Hybrid Physical-Digital Board Games** — camera-based and sensor-based approaches that preserve the tactile experience of rolling physical dice while automating digital game rules.

---

## 2.1 Classical Segmentation, Morphology, and Region Analysis

Because the project deliberately avoids deep learning and pretrained neural networks in favor of transparent, deterministic, real-time processing on CPU, its vision pipeline rests on classical image processing primitives:

- **Global and Adaptive Thresholding:**
  - **Otsu (1979)** introduced nonparametric, unsupervised threshold selection by maximizing inter-class variance in the grayscale histogram. In our pipeline, a white 16 mm die on a dark matte rolling tray produces a strongly bimodal intensity distribution, making Otsu's method ideal for both coarse die segmentation and high-contrast pip extraction inside the cropped die face.
  - **Bradley & Roth (2007)** demonstrated real-time local adaptive thresholding using integral images to handle spatial illumination gradients. When off-axis room lighting introduces mild vignetting across the play area, adaptive thresholding serves as a drop-in alternative to global Otsu thresholding.

- **Mathematical Morphology:**
  - **Serra (1982)** and **Haralick, Sternberg, & Zhuang (1987)** formalized morphological *opening* ($\text{erosion} \circ \text{dilation}$) and *closing* ($\text{dilation} \circ \text{erosion}$).
  - In our pipeline, morphological closing bridges dark pips near the boundary of the white die during die localization so the die body forms a single convex contour, while morphological opening removes isolated sensor noise and specular dust specks inside the cropped die face before pip counting.

- **Contours, Rotated Bounding Boxes, and Connected-Component Labeling:**
  - **Suzuki & Abe (1985)** developed the topological border-following algorithm implemented in OpenCV's `cv2.findContours`, enabling fast extraction of external object boundaries.
  - **Freeman & Shapira (1975)** established minimum-area encasing rectangles for arbitrary closed curves (`cv2.minAreaRect`), which our pipeline uses to estimate the die's planar orientation angle $\theta$ and warp the top face upright.
  - **Rosenfeld & Pfaltz (1966)** and **Wu, Otoo, & Shoshani (2009)** developed two-pass connected-component labeling (`cv2.connectedComponentsWithStats`), providing exact pixel area, centroid, and bounding-box statistics for candidate pips.
  - **Bradski (2000)** (The OpenCV Library) provides `SimpleBlobDetector`, which combines multi-threshold binarization with geometric shape descriptors—**area**, **circularity** ($\frac{4\pi A}{P^2}$), **convexity**, and **inertia ratio**—to separate true circular pips from elongated edge shadows.

---

## 2.2 Automated Dice and Pip Recognition in Machine Vision

Several studies have investigated automatic dice reading for tabletop gaming, casino monitoring, and industrial inspection:

- **Monochrome CCD Pip Extraction in Gaming Tables:**
  - **Correia et al. (1996)** proposed an early online machine vision system using a monochrome CCD camera to locate dice and classify pip scores on gaming tables via thresholding and region counting. While effective under studio lighting, simple region counting is vulnerable to specular glare and edge shadows that mimic or obscure pips.

- **Structural Features and Inter-Pip Distance Clustering:**
  - **Chuang,chen, & Hwang (2009, 2011)** (*"Image automatic-recognition scheme for dice game using structure features"*, *Optical Engineering*, 50(8)) analyzed the structural geometry of dice faces—specifically pip size, circularity filtering ($r \ge 0.85$), morphological opening, and invariant relative inter-pip Euclidean distances—to group pips and identify scores from overhead images.
  - **Relevance to Our Work:** Chuang et al. demonstrated that raw pip counts fail when non-circular artifacts appear and introduced circularity filtering plus structural relationships between pips. Our project advances this idea for a single top-down die by rotating the cropped die face upright using its minimum-area bounding box and mapping detected pip centroids directly onto a **canonical $3 \times 3$ pip layout grid**, verifying that the spatial pattern matches one of the six valid Western die faces.

- **Feature-Invariant & Perspective-Aware Dice Reading:**
  - **Liao et al. (2013)** (*"Multivariate dice recognition using invariant features"*) and **Leng et al. (2024)** (*ISPRS Archives*) highlighted how scale variation, planar rotation, and radial distortion alter apparent pip shapes into ellipses. By restricting the camera to a locked top-down view and rectifying the die crop via its rotated bounding rectangle, our pipeline normalizes planar rotation and scale before applying `SimpleBlobDetector` and $3 \times 3$ grid verification.

---

## 2.3 Selective Classification (Reject Option) and Temporal Consensus

A central premise of this project is that **an undetected or rejected roll is far preferable to a wrong read** that moves a player's token to the wrong snake or ladder.

- **Classification with a Reject Option (Accuracy–Rejection Trade-off):**
  - **Chow (1970)** (*IEEE Transactions on Information Theory*) established the optimal Bayes decision rule with a *reject option*, proving that abstaining from low-confidence predictions systematically reduces the error rate on accepted samples.
  - **El-Yaniv & Wiener (2010)** formalized *selective classification* through the **risk–coverage (or accuracy–rejection) trade-off**, where performance is evaluated jointly by accuracy on accepted inputs and the rejection rate.
  - **Relevance to Our Work:** Standard hobbyist and academic dice readers report raw single-frame accuracy without a rejection mechanism. Our pipeline introduces three explicit verification gates—(1) agreement between `SimpleBlobDetector` and Connected-Component counts, (2) $3 \times 3$ geometric pip layout validity, and (3) multi-frame vote stability—and explicitly evaluates the **accuracy-against-rejection curve**.

- **Temporal Inference and Roll Event Detection in Video Streams:**
  - **Scher, Crandall, & Huttenlocher (2008)** (*"Making Real Games Virtual: Tracking Board Game Pieces"*, *ICPR*) showed that single-frame board game classifiers suffer from transient errors caused by player hands, motion blur, and shadows, and demonstrated that temporal aggregation across consecutive frames eliminates transient misclassifications.
  - **Relevance to Our Work:** Rather than reading a single frame or running continuous inference while the die is bouncing, our **Roll Event Detector** monitors frame-to-frame centroid displacement ($\Delta d_t < \epsilon$ for $N$ consecutive frames) to isolate the exact settled window, and then applies **majority voting** across all settled frames.

---

## 2.4 Tangible and Hybrid Physical-Digital Board Games

Research in pervasive gaming and tangible user interfaces (TUIs) underscores why players value physical dice over digital random number generators (RNGs):

- **Bridging Physical Ritual and Digital Automation:**
  - **Benford, Magerkurth, & Ljungstrand (2005)** (*"Bridging the physical and digital in pervasive gaming"*, *Communications of the ACM*) and **Magerkurth et al. (2004)** (*"Towards the Next Generation of Tabletop Gaming Experiences"*) established that physical dice rolling is an essential social and tactile ritual in board games; replacing it with a button click diminishes player engagement, whereas hybrid tabletop systems combine physical tangibility with automated bookkeeping.

- **Sensor-Embedded Dice vs. Camera-Based Vision:**
  - **Hinske & Langheinrich (2007, 2008)** (*"RFIDice – Augmenting Tabletop Dice with RFID"*, *PerGames*; *"Building RFID-Based Augmented Dice with Perfect Recognition Rates"*, *Fun and Games*) embedded passive RFID tags and metallic shielding inside custom dice to read roll outcomes via an under-table antenna. However, hardware-instrumented dice (RFID or Bluetooth accelerometer dice) require custom manufacturing, alter weight balance, or require battery charging.
  - **Molla & Lepetit (2010)** (*"Augmented Reality for Board Games"*, *IEEE ISMAR*) and **Bérard (2003)** (*"The Magic Table"*, *ICCV Workshop*) demonstrated that a standard overhead webcam combined with classical computer vision can turn unmodified physical game pieces into tangible interfaces for digital game logic.
  - **Relevance to Our Work:** Using an unmodified 16 mm standard white die and an off-the-shelf smartphone camera via USB webcam software achieves zero-cost tangible interaction while keeping the die completely natural and unweighted.

---

## 2.5 Comparative Synthesis: Prior Approaches vs. Proposed System

| Dimension | Hardware-Embedded Dice (Hinske & Langheinrich, 2007–2008) | Standard Single-Frame CV Counting (Correia et al., 1996; Basic Blob Count) | Structural Pip Clustering (Chuang et al., 2009, 2011) | **Proposed Self-Verifying CV Pipeline (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Die Hardware** | Custom RFID-tagged / Bluetooth dice | Standard physical die | Standard physical dice | **Standard 16 mm white physical die** |
| **Sensing Modality** | Under-table RFID antenna array | Monochrome / RGB camera | Overhead RGB camera | **Single top-down smartphone USB camera** |
| **Deep Learning Required?** | No | No | No | **No (100% classical CV, CPU real-time)** |
| **Pip Counting Mechanism** | Tag ID lookup | Threshold + single region counter | Threshold + opening + circularity + distance clustering | **Dual-check: `SimpleBlobDetector` + Connected Components** |
| **Geometric Face Verification** | N/A | None (any $1\text{–}6$ blobs accepted) | Relative inter-pip distance heuristics | **Upright rotation + canonical $3 \times 3$ pip grid matching** |
| **Motion / Roll Handling** | Manual trigger / stationary read | Often reads mid-motion or single snapshot | Static captured images | **Roll Event FSM ($\Delta\text{pos} < \epsilon$ for $N$ frames)** |
| **Temporal Aggregation** | Single read | Single-frame read | Single-frame read | **Multi-frame majority voting over settled window** |
| **Uncertain Read Handling** | Silent failure / missed tag | Misclassification spoils game turn | Exception heuristic for 6-pip face | **Self-verifying reject option (`UNCERTAIN` + re-roll prompt)** |

---

## References

1. **Benford, S., Magerkurth, C., & Ljungstrand, P.** (2005). Bridging the physical and digital in pervasive gaming. *Communications of the ACM*, 48(3), 54–57.
2. **Bérard, F.** (2003). The Magic Table: Computer-vision based augmentation of a whiteboard for creative meetings. In *IEEE International Workshop on Projector-Camera Systems (PROCAMS)*.
3. **Bradley, D., & Roth, G.** (2007). Adaptive thresholding using the integral image. *Journal of Graphics Tools*, 12(2), 13–21.
4. **Bradski, G.** (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, 25(11), 120–123.
5. **Chow, C. K.** (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46.
6. **Chuang, C.-H., Chen, Y.-S., & Hwang, W.-L.** (2009). Image identification scheme for dice game. In *Proceedings of the International Conference on Advanced Information Technologies (AIT)*.
7. **Chuang, C.-H., et al.** (2011). Image automatic-recognition scheme for dice game using structure features. *Optical Engineering*, 50(8), 087005.
8. **Correia, B. A., Davies, R., & Bento, F.** (1996). Automatic detection and classification of dice scores on casino playing tables. *Machine Vision Applications*.
9. **El-Yaniv, R., & Wiener, Y.** (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research (JMLR)*, 11, 1605–1641.
10. **Freeman, H., & Shapira, R.** (1975). Determining the minimum-area encasing rectangle for an arbitrary closed curve. *Communications of the ACM*, 18(7), 409–413.
11. **Haralick, R. M., Sternberg, S. R., & Zhuang, X.** (1987). Image analysis using mathematical morphology. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 9(4), 532–550.
12. **Hinske, S., & Langheinrich, M.** (2007). RFIDice – Augmenting tabletop dice with RFID. In *4th International Symposium on Pervasive Gaming Applications (PerGames)*, 115–122.
13. **Hinske, S., & Langheinrich, M.** (2008). Building RFID-based augmented dice with perfect recognition rates. In *International Conference on Fun and Games* (pp. 59–70). Springer.
14. **Molla, E., & Lepetit, V.** (2010). Augmented reality for board games. In *9th IEEE International Symposium on Mixed and Augmented Reality (ISMAR)*, 253–254.
15. **Otsu, N.** (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
16. **Rosenfeld, A., & Pfaltz, J. L.** (1966). Sequential operations in digital picture processing. *Journal of the ACM*, 13(4), 471–494.
17. **Scher, S., Crandall, D., & Huttenlocher, D.** (2008). Making real games virtual: Tracking board game pieces. In *19th International Conference on Pattern Recognition (ICPR)*, 1–4.
18. **Serra, J.** (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
19. **Suzuki, S., & Abe, K.** (1985). Topological structural analysis of digitized binary images by border following. *Computer Vision, Graphics, and Image Processing*, 30(1), 32–46.
20. **Wu, K., Otoo, E., & Shoshani, A.** (2009). Optimizing two-pass connected-component labeling algorithms. *Pattern Analysis and Applications*, 12(2), 117–135.
