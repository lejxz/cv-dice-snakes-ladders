# 3. Project Details and Tech Stack

---

## 3.1 Project Identity & Overview

- **Project Title:** Self-Verifying Real-Time Dice Recognition Using Blob Detection and Morphology for a Computer Vision-Driven Snakes and Ladders Game
- **Domain:** Classical Computer Vision, Image Processing, Tangible Human-Computer Interaction (HCI)
- **Paradigm:** 100% Classical Computer Vision (No Deep Learning, No Pretrained Neural Networks)

---

## 3.2 Problem Statement, Objective, and Core Contributions

### The Problem
Digital Snakes and Ladders games replace the physical die with a pseudo-random number generator (RNG), stripping away the tactile feel and social anticipation of throwing a real die. Conversely, traditional physical board games require players to read the die and move tokens manually. Bridging the two—allowing players to throw a real die while a computer automatically reads the outcome and advances the digital token—introduces a critical reliability requirement: **a single misread die face immediately spoils the game turn**. Therefore, the vision system must not only read the die in real time, but also **self-verify** whether each read can be trusted.

### Main Objective
Build a real-time classical computer vision system that:
1. Watches a physical white 16 mm die through a single top-down smartphone or USB camera.
2. Detects when a thrown die has come to a complete stop (settled).
3. Reads the top face (pips $1$ to $6$) using classical segmentation, morphology, blob detection, and connected-component analysis.
4. Self-verifies the read via dual-method agreement, $3 \times 3$ geometric pip layout matching, and multi-frame majority voting.
5. Drives a minimal digital Snakes and Ladders board (~150 lines of game logic) with an annotated live camera feed and manual override fallback.

### Core Contributions
While the underlying operators (thresholding, mathematical morphology, `SimpleBlobDetector`, connected components) are established, this project contributes a cohesive **self-verification framework** evaluated via rigorous ablation studies:
1. **Geometric $3 \times 3$ Pip Layout Verification:** Detected pip centroids on the upright-warped die face are mapped onto a canonical $3 \times 3$ grid and must match one of the six valid Western die face configurations.
2. **Multi-Frame Majority Voting Over Settled Frames:** Replaces brittle single-frame snapshots by aggregating predictions across all consecutive settled frames of a roll event.
3. **Roll Event Detection:** Temporal centroid tracking detects the exact transition from motion to rest ($\Delta\text{pos} < \epsilon$ for $N$ consecutive frames), preventing premature reads on moving or bouncing dice.
4. **Selective Classification Reporting (Accuracy vs. Rejection Rate):** Evaluates accuracy jointly with the rejection rate (`UNCERTAIN` reads), quantifying the reliability gained by refusing ambiguous reads.

### Scope & Assumptions
- **Classical CV Only:** Strictly uses deterministic pixel/region algorithms (Otsu/adaptive thresholding, morphological opening/closing, contour geometry, `SimpleBlobDetector`, `connectedComponentsWithStats`).
- **Single White Die on Dark Matte Surface:** A 16 mm standard white die with dark pips is rolled onto a plain, dark, matte play surface under fixed, diffuse lighting.
- **Flat Landing Assumption:** The die is assumed to land flat within the designated region of interest (ROI).
- **Minimal Game Logic:** The Snakes and Ladders engine is intentionally concise (~150 LOC: token movement, ladder climbs, snake slides, bonus roll on 6) so that computer vision remains the primary focus.

---

## 3.3 Hardware & Imaging Setup

| Component | Specification / Configuration | Rationale |
| :--- | :--- | :--- |
| **Camera Sensor** | Android Smartphone (via USB webcam app) or Dedicated USB Webcam | Accessible, high-definition sensor with zero specialized hardware cost. |
| **Camera Mounting** | Overhead tripod / gooseneck mount facing straight down ($90^\circ$ nadir) | Minimizes perspective foreshortening and keeps side faces of the die hidden. |
| **Connection** | Wired USB 2.0/3.0 connection to development laptop | Eliminates Wi-Fi packet loss, compression artifacts, and latency spikes. |
| **Resolution & Frame Rate** | Up to $1920 \times 1080$ (minimum $1280 \times 720$) at $\approx 30\text{ FPS}$ | High resolution preserves small pip geometry; 30 FPS enables responsive roll-settle tracking. |
| **Camera Controls** | Locked Exposure, Locked Focus, Locked White Balance | Prevents auto-exposure pumping or focus hunting when a player's hand enters the frame. |
| **Physical Die** | Standard 16 mm opaque white cube die with high-contrast dark pips | Provides strong bimodal intensity contrast for both die body and pips. |
| **Rolling Surface** | Plain, dark, matte surface (e.g., black neoprene mat or felt tray) | Suppresses background reflections and maximizes foreground-background contrast. |
| **Lighting** | Fixed, diffuse overhead lighting | Minimizes harsh directional shadows and specular glare on the glossy die face. |

---

## 3.4 Software Tech Stack

### Core Runtime Stack

| Layer | Technology | Version | Purpose in Pipeline |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.10+` | Primary implementation language for vision, game logic, and evaluation scripts. |
| **Computer Vision** | OpenCV (`opencv-python`) | `4.8+` | Video capture, grayscale/blur, Otsu/adaptive thresholding, morphology (`morphologyEx`), contours (`findContours`, `minAreaRect`), affine warping (`warpAffine`), `SimpleBlobDetector`, and `connectedComponentsWithStats`. |
| **Numerical Computing** | NumPy | `1.24+` | Fast 2D array slicing, $3 \times 3$ pip grid matrix comparison, Euclidean centroid distances, and voting statistics. |
| **Game UI & Windowing** | PyGame (`pygame-ce` / `pygame`) | `2.5+` | Renders the digital Snakes and Ladders board, animated tokens, side-by-side annotated camera feed, and keyboard event loop (manual override keys). |
| **Phone-as-Webcam Bridge** | DroidCam / Iriun Webcam / Scrcpy (V4L2 / DirectShow) | Latest | Exposes the USB-connected Android smartphone camera as a standard OS webcam device index for `cv2.VideoCapture`. |

### Evaluation, Data & Testing Stack

| Tool / Library | Purpose |
| :--- | :--- |
| **Pandas & CSV (`pandas`, `csv`)** | Logging per-roll predictions, confidence scores, frame latencies, and comparing against hand-recorded ground truth (`ground_truth.csv`). |
| **Matplotlib & Seaborn (`matplotlib`, `seaborn`)** | Generating confusion matrices, accuracy-vs-rejection curves, ablation bar charts, and error analysis figures. |
| **Scikit-Learn (`scikit-learn`)** | Computing multi-class confusion matrices, precision/recall per face, and selective classification curves. |
| **Pytest (`pytest`)** | Unit testing $3 \times 3$ grid template matching, roll-event state machine transitions, and Snakes and Ladders board rules. |

---

## 3.5 Proposed Repository Structure

```text
cv-dice-snakes-ladders/
├── concept/
│   ├── README.md                              # Concept overview and navigation index
│   ├── 01-implementation-plan.md              # Multi-step implementation roadmap
│   ├── 02-related-literature.md               # Academic literature review & synthesis
│   └── 03-project-details-and-tech-stack.md   # Project specs, hardware, and software stack
├── src/
│   ├── __init__.py
│   ├── config.py                              # Tunable CV parameters (ROI, thresholds, N frames)
│   ├── acquisition/
│   │   ├── camera.py                          # Live USB camera & recorded clip reader
│   │   └── recorder.py                        # Utility to record roll clips for evaluation
│   ├── vision/
│   │   ├── preprocessing.py                   # Grayscale, Gaussian blur, dual-resolution scaling
│   │   ├── die_detector.py                    # Otsu/adaptive threshold, morphology, rotated rect crop
│   │   ├── roll_tracker.py                    # Roll event FSM (moving -> settling -> settled)
│   │   ├── pip_detector.py                    # SimpleBlobDetector + Connected Components
│   │   └── verifier.py                        # 3x3 pip grid check + multi-frame majority voting
│   ├── game/
│   │   └── board.py                           # Minimal ~150-LOC Snakes and Ladders engine
│   ├── ui/
│   │   └── app.py                             # Unified OpenCV + PyGame window & manual overrides
│   └── evaluation/
│       ├── benchmark.py                       # Batch evaluation & ablation runner (>=150 rolls)
│       └── plots.py                           # Confusion matrix & accuracy-rejection curve generator
├── data/
│   ├── recordings/                            # Recorded roll videos (.mp4)
│   └── ground_truth.csv                       # Hand-recorded true values for evaluation
├── tests/                                     # Unit tests for vision verification and game rules
├── reports/                                   # Generated metrics, plots, and error analysis images
├── requirements.txt
└── README.md
```

---

## 3.6 Target Performance Metrics & Evaluation Protocol

### Quantitative Targets

| Metric | Target Threshold | Measurement Protocol |
| :--- | :--- | :--- |
| **Pip-Read Accuracy (Settled Dice)** | **$\ge 95\%$** | Measured over $\ge 150$ physical rolls across faces 1–6, reported jointly with the rejection rate. |
| **Real-Time Throughput (FPS)** | **$\ge 15\text{ FPS}$** | End-to-end loop (acquisition + CV pipeline + UI rendering) measured on the development laptop. |
| **Roll Event Reliability** | Near-zero premature reads | Count of premature reads (reading while die is still moving) and missed settles (failing to trigger when stopped). |

### Required Ablation & Comparative Experiments
1. **Effect of $3 \times 3$ Pip Layout Verification:** Compare accuracy and false-acceptance rate *with* vs. *without* the $3 \times 3$ geometric grid check.
2. **Effect of Multi-Frame Majority Voting:** Compare multi-frame majority voting across all settled frames against a single-frame read (first settled frame).
3. **Blob Detection vs. Connected Components:** Compare `SimpleBlobDetector` alone, `connectedComponentsWithStats` alone, and their dual-agreement combination.
4. **Accuracy–Rejection Trade-off Curve:** Plot system accuracy as a function of rejection rate by varying the verification strictness / confidence threshold.
5. **Environmental Robustness & Failure Mode Analysis:** Evaluate across varying illumination levels and die positions within the play area, accompanied by annotated visual examples of the most common failure cases.

---

## 3.7 Intended Users & Applications

1. **Families and Casual Tabletop Players:** Enables a hybrid physical-digital board game experience where players enjoy tactile physical dice throws while the computer handles token movement, snakes, and ladders automatically.
2. **STEM Educators and Students:** Serves as an interactive classroom demonstration of empirical probability distributions (Law of Large Numbers over real dice throws) and a transparent, glass-box introduction to classical computer vision (thresholding, morphology, blobs, and geometric verification).
3. **Accessibility & Ergonomics-Focused Players:** Offers players who prefer throwing a physical object over clicking a mouse a responsive, self-checking tabletop interface with a simple keyboard override fallback.
