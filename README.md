# Motion Segmentation (MOG2 & GMG)

## 📌 Overview
This project explores **image processing**, **video processing**, and **video tracking** techniques using OpenCV.  
It focuses on background subtraction algorithms (**MOG2** and **GMG**) combined with **morphological operations** to reduce noise and improve segmentation quality.  
The implementation includes a unified GUI that displays the original video, raw mask, and processed mask side by side for easy comparison.

---

## 🎯 Objectives
- Implement **background subtraction** using MOG2 and GMG.
- Apply **morphological operations** (erosion, dilation, opening, closing) to clean binary masks.
- Provide a **real-time video GUI** with three panels:
  - Original frame
  - Raw mask
  - Morphologically processed mask
- Compare the performance and robustness of MOG2 vs GMG in dynamic environments.

---

## 🛠 Features
- **MOG2 Background Subtraction**: robust, fast, and suitable for real-time tracking.
- **GMG Background Subtraction**: probabilistic model, sensitive to texture and light variations.
- **Morphological Noise Reduction**:
  - `erode` → removes small noise
  - `dilate` → fills gaps
  - `open` → erosion followed by dilation (noise removal)
  - `close` → dilation followed by erosion (object consolidation)
- **Interactive GUI**: single window split into three sections with labels for clarity.
- **Configurable Parameters**: history length, thresholds, shadow detection, initialization frames, decision thresholds.

---
