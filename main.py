
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_ROOT = "/kaggle/input/kitti-3d-object-detection-dataset/training"



print("Images:", len(os.listdir(os.path.join(DATA_ROOT, "image_2"))))
print("LiDAR:", len(os.listdir(os.path.join(DATA_ROOT, "velodyne_reduced"))))
print("Calib:", len(os.listdir(os.path.join(DATA_ROOT, "calib"))))

img_path = os.path.join(DATA_ROOT, "image_2", "000123.png")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,4))
plt.imshow(img)
plt.axis("off")
plt.title("KITTI Camera Image")


lidar_path = os.path.join(DATA_ROOT, "velodyne_reduced", "000123.bin")
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

print("Point cloud shape:", points.shape)
print("XYZ min:", points[:, :3].min(axis=0))
print("XYZ max:", points[:, :3].max(axis=0))


plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], s=0.1)
plt.xlabel("X (forward)")
plt.ylabel("Y (left)")
plt.title("LiDAR BEV")
plt.axis("equal")


import numpy as np

def load_kitti_calib(calib_path):
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
    return calib


calib_path = "/kaggle/input/kitti-3d-object-detection-dataset/training/calib/000123.txt"
calib = load_kitti_calib(calib_path)

P2 = calib['P2'].reshape(3, 4)                  # Camera projection
Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
R0_rect = calib['R0_rect'].reshape(3, 3)


Tr_velo_to_cam_h = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
R0_rect_h = np.eye(4)
R0_rect_h[:3, :3] = R0_rect


lidar_path = "/kaggle/input/kitti-3d-object-detection-dataset/training/velodyne_reduced/000123.bin"
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

# Homogeneous coordinates
points_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])

# LiDAR → Camera
points_cam = (R0_rect_h @ Tr_velo_to_cam_h @ points_h.T).T


mask = points_cam[:, 2] > 0
points_cam = points_cam[mask]


points_img = (P2 @ points_cam[:, :4].T).T

points_img[:, 0] /= points_img[:, 2]
points_img[:, 1] /= points_img[:, 2]


import cv2
import matplotlib.pyplot as plt

img_path = "/kaggle/input/kitti-3d-object-detection-dataset/training/image_2/000123.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

for x, y, z in zip(points_img[:,0], points_img[:,1], points_cam[:,2]):
    if 0 <= x < w and 0 <= y < h:
        color = plt.cm.jet(min(z / 50, 1.0))[:3]
        img[int(y), int(x)] = (np.array(color) * 255).astype(np.uint8)

plt.figure(figsize=(12,5))
plt.imshow(img)
plt.axis("off")
plt.title("LiDAR Points Projected onto Camera Image")


def load_kitti_labels(label_path):
    objects = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            obj = {
                "type": data[0],
                "bbox": list(map(float, data[4:8]))  # xmin, ymin, xmax, ymax
            }
            objects.append(obj)
    return objects


def lidar_points_in_box(points_img, points_cam, bbox):
    xmin, ymin, xmax, ymax = bbox
    mask = (
        (points_img[:,0] >= xmin) &
        (points_img[:,0] <= xmax) &
        (points_img[:,1] >= ymin) &
        (points_img[:,1] <= ymax)
    )
    return points_cam[mask]


def estimate_object_depth(points_cam):
    if len(points_cam) == 0:
        return None
    return np.median(points_cam[:, 2])  # Z = forward distance


label_path = "/kaggle/input/kitti-3d-object-detection-dataset/training/label_2/000123.txt"
objects = load_kitti_labels(label_path)

object_depths = []

for obj in objects:
    if obj["type"] not in ["Car", "Pedestrian", "Cyclist"]:
        continue

    pts_in_box = lidar_points_in_box(points_img, points_cam, obj["bbox"])
    depth = estimate_object_depth(pts_in_box)

    object_depths.append({
        "type": obj["type"],
        "bbox": obj["bbox"],
        "depth": depth
    })


img_vis = img.copy()

for obj in object_depths:
    if obj["depth"] is None:
        continue

    xmin, ymin, xmax, ymax = map(int, obj["bbox"])
    depth = obj["depth"]

    cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    cv2.putText(
        img_vis,
        f"{obj['type']} {depth:.1f}m",
        (xmin, ymin-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,0,0),
        1
    )

plt.figure(figsize=(12,5))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Object-Level Metric Depth from Camera–LiDAR Fusion")


class KalmanTrack:
    def __init__(self, x, z, track_id):
        self.id = track_id
        self.x = np.array([x, z, 0.0, 0.0])  # initial velocity = 0
        self.P = np.eye(4) * 10.0

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Q = np.eye(4) * 0.1

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_meas):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * 1.0

        y = z_meas - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


def bbox_center_to_3d(bbox, depth, P2):
    xmin, ymin, xmax, ymax = bbox
    u = (xmin + xmax) / 2
    v = (ymin + ymax) / 2

    fx = P2[0,0]
    fy = P2[1,1]
    cx = P2[0,2]
    cy = P2[1,2]

    X = (u - cx) * depth / fx
    Z = depth
    return X, Z


def associate_detections(tracks, measurements, max_dist=3.0):
    associations = {}
    used = set()

    for i, trk in enumerate(tracks):
        best_j, best_d = None, float("inf")
        for j, meas in enumerate(measurements):
            if j in used:
                continue
            d = np.linalg.norm(trk.x[:2] - meas)
            if d < best_d and d < max_dist:
                best_d = d
                best_j = j

        if best_j is not None:
            associations[i] = best_j
            used.add(best_j)

    unmatched = [j for j in range(len(measurements)) if j not in used]
    return associations, unmatched


tracks = []
next_id = 0
dt = 0.1  # KITTI ~10 Hz

for frame_idx in range(123, 130):  # small sequence
    # Load labels, LiDAR, projection (reuse earlier code)

    measurements = []
    for obj in object_depths:
        if obj["depth"] is None:
            continue
        X, Z = bbox_center_to_3d(obj["bbox"], obj["depth"], P2)
        measurements.append(np.array([X, Z]))

    # Predict
    for trk in tracks:
        trk.predict(dt)

    # Associate
    assoc, unmatched = associate_detections(tracks, measurements)

    # Update matched tracks
    for ti, mi in assoc.items():
        tracks[ti].update(measurements[mi])

    # Create new tracks
    for mi in unmatched:
        trk = KalmanTrack(measurements[mi][0], measurements[mi][1], next_id)
        tracks.append(trk)
        next_id += 1


img_trk = img.copy()

for trk in tracks:
    X, Z, Vx, Vz = trk.x
    u = int((X * P2[0,0] / Z) + P2[0,2])
    v = int(P2[1,2])

    cv2.putText(
        img_trk,
        f"ID {trk.id} | Vz={Vz:.1f} m/s",
        (u, v),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,0,0),
        1
    )

plt.figure(figsize=(12,5))
plt.imshow(img_trk)
plt.axis("off")
plt.title("Multi-Object Tracking with Velocity Estimation")


def compute_ttc(Z, Vz, min_ttc=0.1, max_ttc=20.0):
    if Vz >= 0:      # not approaching
        return None
    ttc = Z / (-Vz)
    if ttc < min_ttc or ttc > max_ttc:
        return None
    return ttc


def ttc_risk_level(ttc):
    if ttc is None:
        return None
    if ttc < 2.0:
        return "HIGH"
    elif ttc < 5.0:
        return "MEDIUM"
    else:
        return "LOW"


img_ttc = img.copy()

for trk in tracks:
    X, Z, Vx, Vz = trk.x
    ttc = compute_ttc(Z, Vz)
    risk = ttc_risk_level(ttc)

    if risk is None:
        continue

    # Project track position back to image
    u = int((X * P2[0,0] / Z) + P2[0,2])
    v = int(P2[1,2])

    color = {
        "LOW": (0,255,0),
        "MEDIUM": (255,165,0),
        "HIGH": (255,0,0)
    }[risk]

    text = f"ID {trk.id} | TTC={ttc:.1f}s | {risk}"

    cv2.putText(
        img_ttc,
        text,
        (u, v),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

plt.figure(figsize=(12,5))
plt.imshow(img_ttc)
plt.axis("off")
plt.title("ADAS Time-to-Collision Risk Visualization")


def process_frame(frame_id, tracks, next_id, dt):
    # ---- Load data ----
    img_path = f"{DATA_ROOT}/image_2/{frame_id:06d}.png"
    lidar_path = f"{DATA_ROOT}/velodyne_reduced/{frame_id:06d}.bin"
    calib_path = f"{DATA_ROOT}/calib/{frame_id:06d}.txt"
    label_path = f"{DATA_ROOT}/label_2/{frame_id:06d}.txt"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    calib = load_kitti_calib(calib_path)

    P2 = calib['P2'].reshape(3,4)
    Tr = calib['Tr_velo_to_cam'].reshape(3,4)
    R0 = calib['R0_rect'].reshape(3,3)

    Tr_h = np.vstack([Tr, [0,0,0,1]])
    R0_h = np.eye(4); R0_h[:3,:3] = R0

    pts_h = np.hstack([points[:,:3], np.ones((points.shape[0],1))])
    pts_cam = (R0_h @ Tr_h @ pts_h.T).T
    pts_cam = pts_cam[pts_cam[:,2] > 0]

    pts_img = (P2 @ pts_cam[:,:4].T).T
    pts_img[:,0] /= pts_img[:,2]
    pts_img[:,1] /= pts_img[:,2]

    # ---- Load objects ----
    objects = load_kitti_labels(label_path)

    measurements = []
    for obj in objects:
        if obj["type"] not in ["Car", "Pedestrian", "Cyclist"]:
            continue
        pts_in = lidar_points_in_box(pts_img, pts_cam, obj["bbox"])
        depth = estimate_object_depth(pts_in)
        if depth is None:
            continue
        X, Z = bbox_center_to_3d(obj["bbox"], depth, P2)
        measurements.append(np.array([X, Z]))

    # ---- Predict ----
    for trk in tracks:
        trk.predict(dt)

    # ---- Associate ----
    assoc, unmatched = associate_detections(tracks, measurements)

    # ---- Update ----
    for ti, mi in assoc.items():
        tracks[ti].update(measurements[mi])

    # ---- New tracks ----
    for mi in unmatched:
        tracks.append(KalmanTrack(
            measurements[mi][0],
            measurements[mi][1],
            next_id
        ))
        next_id += 1

    return img, tracks, next_id


tracks = []
next_id = 0
dt = 0.1  # KITTI ~10 Hz

frame_ids = range(120, 140)

for fid in frame_ids:
    img, tracks, next_id = process_frame(fid, tracks, next_id, dt)

    img_vis = img.copy()

    for trk in tracks:
        X, Z, Vx, Vz = trk.x
        ttc = compute_ttc(Z, Vz)
        risk = ttc_risk_level(ttc)

        if risk is None:
            continue

        u = int((X * P2[0,0] / Z) + P2[0,2])
        v = int(P2[1,2])

        color = {"LOW":(0,255,0),"MEDIUM":(255,165,0),"HIGH":(255,0,0)}[risk]

        cv2.putText(
            img_vis,
            f"ID {trk.id} TTC={ttc:.1f}s",
            (u, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    plt.figure(figsize=(10,4))
    plt.imshow(img_vis)
    plt.axis("off")
    plt.title(f"Frame {fid}")
    plt.show()




tracks = []
next_id = 0
dt = 0.1  # KITTI ~10 Hz

frame_ids = range(120, 140)

for fid in frame_ids:
    img, tracks, next_id = process_frame(fid, tracks, next_id, dt)
    
    img_vis = img.copy()
    
    for trk in tracks:
        X, Z, Vx, Vz = trk.x
    
        u = int((X * P2[0,0] / Z) + P2[0,2])
        v = int(P2[1,2])
    
        cv2.putText(
            img_vis,
            f"ID {trk.id} Z={Z:.1f}m Vz={Vz:.1f}",
            (u, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )
    
    plt.figure(figsize=(10,4))
    plt.imshow(img_vis)
    plt.axis("off")
    plt.title(f"Frame {fid}")
    plt.show()





def depth_error(z_pred, z_gt):
    return abs(z_pred - z_gt)


def load_kitti_labels(label_path):
    objects = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()

            if data[0] == "DontCare":
                continue

            obj = {
                "type": data[0],
                "bbox": list(map(float, data[4:8])),
                "gt_z": float(data[13])
            }
            objects.append(obj)
    return objects


label_path = "/kaggle/input/kitti-3d-object-detection-dataset/training/label_2/000123.txt"
objects = load_kitti_labels(label_path)


for obj in objects[:5]:
    print(obj["type"], "GT depth:", obj["gt_z"])


DATA_ROOT = "/kaggle/input/kitti-3d-object-detection-dataset/training"


def project_lidar_to_image(points, calib):
    P2 = calib['P2'].reshape(3,4)
    Tr = calib['Tr_velo_to_cam'].reshape(3,4)
    R0 = calib['R0_rect'].reshape(3,3)

    Tr_h = np.vstack([Tr, [0,0,0,1]])
    R0_h = np.eye(4); R0_h[:3,:3] = R0

    pts_h = np.hstack([points[:,:3], np.ones((points.shape[0],1))])
    pts_cam = (R0_h @ Tr_h @ pts_h.T).T
    pts_cam = pts_cam[pts_cam[:,2] > 0]

    pts_img = (P2 @ pts_cam[:,:4].T).T
    pts_img[:,0] /= pts_img[:,2]
    pts_img[:,1] /= pts_img[:,2]

    return pts_img, pts_cam


frame_ids = range(120, 220)  # ~100 frames


import os
import numpy as np

depth_errors = []        # list of absolute depth errors
depth_pairs = []         # (z_gt, abs_error)
skipped_objects = 0
total_objects = 0


frame_ids = range(120, 220)   # ~100 frames


for fid in frame_ids:
    label_path = f"{DATA_ROOT}/label_2/{fid:06d}.txt"
    lidar_path = f"{DATA_ROOT}/velodyne_reduced/{fid:06d}.bin"
    calib_path = f"{DATA_ROOT}/calib/{fid:06d}.txt"

    if not (os.path.exists(label_path) and os.path.exists(lidar_path)):
        continue

    objects = load_kitti_labels(label_path)
    if len(objects) == 0:
        continue

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    calib = load_kitti_calib(calib_path)

    pts_img, pts_cam = project_lidar_to_image(points, calib)

    for obj in objects:
        if obj["type"] not in ["Car", "Truck", "Pedestrian", "Cyclist"]:
            continue

        total_objects += 1

        pts_in = lidar_points_in_box(pts_img, pts_cam, obj["bbox"])
        z_pred = estimate_object_depth(pts_in)

        if z_pred is None:
            skipped_objects += 1
            continue

        z_gt = obj["gt_z"]
        error = abs(z_pred - z_gt)

        depth_errors.append(error)
        depth_pairs.append((z_gt, error))


depth_errors = np.array(depth_errors)


depth_errors = np.array(depth_errors)

print("Depth evaluation summary")
print("------------------------")
print(f"Frames evaluated   : {len(list(frame_ids))}")
print(f"Objects evaluated  : {len(depth_errors)}")
print(f"Objects skipped    : {skipped_objects}")
print(f"Depth MAE (mean)   : {depth_errors.mean():.2f} m")
print(f"Depth median error : {np.median(depth_errors):.2f} m")
print(f"Depth 90th percentile: {np.percentile(depth_errors, 90):.2f} m")


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(
    [p[0] for p in depth_pairs],
    [p[1] for p in depth_pairs],
    s=5
)
plt.xlabel("Ground Truth Depth (m)")
plt.ylabel("Absolute Depth Error (m)")
plt.title("Depth Error vs Distance (KITTI)")
plt.grid(True)
plt.show()
