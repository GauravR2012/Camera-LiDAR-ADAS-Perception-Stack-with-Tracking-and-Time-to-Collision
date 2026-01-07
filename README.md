# Camera–LiDAR ADAS Perception Stack (KITTI)

## Overview
This project implements an **end-to-end ADAS perception pipeline** using **camera–LiDAR sensor fusion** on the KITTI dataset.  
The system estimates metric depth, tracks road agents over time, computes relative velocity, and derives **Time-to-Collision (TTC)** for forward collision risk assessment.

The goal of this project is to demonstrate **real-world robotics perception skills**, including multi-sensor geometry, state estimation, and safety-oriented logic.

---

## System Architecture

Camera Image + LiDAR Point Cloud
↓
Camera–LiDAR Calibration & Projection
↓
Object-Level LiDAR Association
↓
Metric Depth Estimation
↓
Multi-Object Tracking (Kalman Filter)
↓
Velocity Estimation
↓
Time-to-Collision (ADAS Risk)



---

## Dataset
- Dataset: :contentReference[oaicite:0]{index=0} (3D Object Detection)
- Sensors:
  - Monocular RGB camera
  - Velodyne LiDAR
- Annotations:
  - 2D bounding boxes
  - 3D object positions in camera coordinates

Only the official KITTI `training` split is used for evaluation.

---

## Core Components

### 1. Camera–LiDAR Fusion
- LiDAR points are transformed from the LiDAR frame to the camera frame using KITTI calibration
- Points are projected onto the image plane
- LiDAR points are associated with 2D object bounding boxes

This enables **metric depth estimation** from fused sensor data.

---

### 2. Depth Estimation
- Object depth is computed as the **median LiDAR Z value** inside each 2D bounding box
- Median depth is used for robustness against sparse points and outliers
- Depth is expressed in **meters**, not relative scale

---

### 3. Multi-Object Tracking
- Each object is tracked using a **Kalman Filter**
- State vector:
[X, Z, Vx, Vz]

- Nearest-neighbor data association is used for matching detections to tracks
- Stable IDs are maintained across frames

---

### 4. Time-to-Collision (TTC)
Time-to-Collision is computed for approaching objects as:

\[
TTC = \frac{Z}{-V_z}
\]

- TTC is only computed when an object is approaching the ego vehicle
- Conservative gating is applied to reduce false collision warnings
- TTC is visualized with risk levels (LOW / MEDIUM / HIGH)

---

## Evaluation

### Depth Accuracy (KITTI Ground Truth)

Depth estimation is evaluated against **KITTI 3D ground-truth camera coordinates**.

**Evaluation Setup**
- Frames evaluated: 100
- Objects evaluated: 515
- Metric: Absolute depth error (meters)

**Results**

| Metric | Value |
|------|------|
| Mean Absolute Error (MAE) | **2.76 m** |
| Median Absolute Error | **1.40 m** |
| 90th Percentile Error | **6.56 m** |

**Discussion**
- Low median error indicates accurate depth estimation for most objects
- Larger errors occur for distant objects (>60 m) due to sparse LiDAR returns
- Accuracy is sufficient for ADAS tasks such as tracking and TTC computation

---

## Key Takeaways
- Demonstrates real-world **camera–LiDAR sensor fusion**
- Implements **temporal state estimation** using Kalman filtering
- Converts perception outputs into **driving-relevant safety metrics**
- Designed with **ADAS conservatism** to avoid false collision warnings

---

## Tools & Technologies
- Python
- NumPy
- OpenCV
- Kalman Filtering
- KITTI Dataset

---

## Limitations & Future Work
- Replace ground-truth 2D boxes with a real-time object detector
- Improve long-range depth robustness
- Extend TTC logic with temporal smoothing
- Add lane-level risk reasoning

---

## Conclusion
This project demonstrates a **production-style ADAS perception pipeline**, covering sensor fusion, tracking, and safety-critical reasoning.  
The implementation and evaluation are aligned with real-world autonomous driving and robotics perception systems.


