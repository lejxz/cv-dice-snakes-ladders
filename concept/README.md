# Project Concept & Planning Documentation

**Project Title:** *Self-Verifying Real-Time Dice Recognition Using Blob Detection and Morphology for a Computer Vision-Driven Snakes and Ladders Game*

This folder contains the foundational concept, literature review, architectural specifications, and multi-step implementation roadmap for the project.

---

## Contents

1. **[1. Implementation Plan (`01-implementation-plan.md`)](./01-implementation-plan.md)**
   - Separates the implementation into **8 sequential, testable steps**:
     - **Step 1:** Environment Setup, Video Acquisition & Play-Area Calibration
     - **Step 2:** Dual-Resolution Preprocessing Module
     - **Step 3:** Classical Die Detection & Upright Warping
     - **Step 4:** Roll Event Detection (Temporal State Machine)
     - **Step 5:** Pip Segmentation & Dual-Method Counting (`SimpleBlobDetector` + Connected Components)
     - **Step 6:** Geometric $3 \times 3$ Pip Layout Verification & Multi-Frame Majority Voting
     - **Step 7:** Minimal Snakes and Ladders Game Engine (~150 LOC) & Unified OpenCV/PyGame GUI
     - **Step 8:** Comprehensive Evaluation & Ablation Suite ($\ge 150$ Rolls, Accuracy-Rejection Curve, FPS)

2. **[2. Related Literature (`02-related-literature.md`)](./02-related-literature.md)**
   - Reviews foundational classical CV operators (Otsu & adaptive thresholding, mathematical morphology, contour analysis, connected components, blob detection).
   - Examines prior automated dice recognition pipelines and structural pip geometry methods.
   - Connects selective classification (classification with a reject option) and temporal consensus to our self-verifying pipeline.
   - Situates the work within tangible and hybrid physical-digital board game interfaces.

3. **[3. Project Details and Tech Stack (`03-project-details-and-tech-stack.md`)](./03-project-details-and-tech-stack.md)**
   - Problem statement, main objective, core contributions, and classical-CV constraints.
   - Physical hardware & imaging setup (top-down smartphone USB camera, 16 mm white die, dark matte surface, locked exposure/focus/WB).
   - Software tech stack (Python, OpenCV, NumPy, PyGame, Pandas, Matplotlib, Scikit-Learn, Pytest).
   - Target evaluation metrics ($\ge 95\%$ accuracy on settled dice, $\ge 15\text{ FPS}$, ablation studies) and intended users.
