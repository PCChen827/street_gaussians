import os
import sys
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def visualize_depth_numpy(depth, min_depth=None, max_depth=None):
    x = depth.copy()
    valid_mask = x > 0
    if valid_mask.sum() == 0:
        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8), None
    if min_depth is None: min_depth = x[valid_mask].min()
    if max_depth is None: max_depth = x[valid_mask].max()
    x = (x - min_depth) / (max_depth - min_depth + 1e-5)
    x = np.clip(x, 0, 1)
    x[~valid_mask] = 0
    x_uint8 = (x * 255).astype(np.uint8)
    colored = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
    colored[~valid_mask] = 0
    return colored, None

image_filename_to_cam = lambda x: int(x.split('.')[0].split('_')[-1]) # Format: {frame}_{cam}.png
image_filename_to_frame = lambda x: int(x.split('.')[0].split('_')[0])

def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    
    intrinsics = {}
    extrinsics = {}
    
    # Read camera parameters (6 cameras)
    for i in range(6):
        int_path = os.path.join(intrinsics_dir, f"{i}.txt")
        ext_path = os.path.join(extrinsics_dir, f"{i}.txt")
        
        if os.path.exists(int_path) and os.path.exists(ext_path):
            # Intrinsic
            K = np.loadtxt(int_path)
            if K.shape != (3, 3):
                if len(K.flatten()) == 9:
                     K = K.reshape(3,3)
                else:
                     # fx, fy, cx, cy
                     fx, fy, cx, cy = K[0], K[1], K[2], K[3]
                     K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intrinsics[i] = K
            
            # Extrinsic (Cam -> Ego)
            ext = np.loadtxt(ext_path)
            extrinsics[i] = ext
            
    return extrinsics, intrinsics

def generate_lidar_depth(datadir):
    save_dir = os.path.join(datadir, 'lidar_depth')
    os.makedirs(save_dir, exist_ok=True)
    
    image_dir = os.path.join(datadir, 'images')
    image_files = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))
    
    print(f"Loading pointcloud from {datadir}...")
    pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
    
    if not os.path.exists(pointcloud_path):
        print(f"Error: {pointcloud_path} does not exist.")
        return

    data = np.load(pointcloud_path, allow_pickle=True)
    pts3d_dict = data['pointcloud'].item()
    
    extrinsics, intrinsics = load_calibration(datadir)
    
    print("Generating depth maps...")
    for image_filename in tqdm(image_files):
        image = cv2.imread(image_filename)
        h, w = image.shape[:2]
        
        image_basename = os.path.basename(image_filename)
        try:
            frame = image_filename_to_frame(image_basename)
            cam = image_filename_to_cam(image_basename)
        except:
            print(f"Skipping filename format error: {image_basename}")
            continue

        if frame not in pts3d_dict:
            continue
        
        # Output path
        depth_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.npy')
        depth_vis_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.png')
        
        # Obtain the Ego Frame point cloud of the frame
        raw_3d = pts3d_dict[frame] # (N, 3)
        if raw_3d.shape[0] == 0:
            continue

        # Transformation: Ego -> Camera
        # P_cam = T_ego2cam @ P_ego
        # Extrinsic file: Cam -> Ego (T_cam2ego)
        T_cam2ego = extrinsics[cam]
        T_ego2cam = np.linalg.inv(T_cam2ego)
        
        points_xyz = np.concatenate([raw_3d, np.ones_like(raw_3d[..., :1])], axis=-1) # (N, 4)
        points_cam = points_xyz @ T_ego2cam.T # (N, 4)
        
        # P_img = K @ P_cam_3d
        points_cam_3d = points_cam[..., :3]
        depth_values = points_cam_3d[..., 2]
        
        # Keep front view (Z > 0.1)
        valid_z = depth_values > 0.1
        points_cam_3d = points_cam_3d[valid_z]
        depth_values = depth_values[valid_z]
        
        K = intrinsics[cam]
        points_img = points_cam_3d @ K.T
        points_img = points_img[..., :2] / points_img[..., 2:3] # Divide by Z -> (u, v)
        
        u = points_img[:, 0].round().astype(np.int32)
        v = points_img[:, 1].round().astype(np.int32)
        
        valid_pixel = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid_pixel]
        v = v[valid_pixel]
        d = depth_values[valid_pixel]
        
        depth_map = np.full((h, w), np.inf, dtype=np.float32)
        
        flat_indices = v * w + u
        depth_flat = depth_map.reshape(-1)
        np.minimum.at(depth_flat, flat_indices, d)
        depth_map = depth_flat.reshape(h, w)
        
        mask = depth_map < 1e5
        final_depth_values = np.zeros_like(depth_map)
        final_depth_values[mask] = depth_map[mask]
        
        # Save
        # Format: dict(mask=bool_array, value=float_array_of_valid_pixels)
        depth_file = dict()
        depth_file['mask'] = mask
        depth_file['value'] = final_depth_values[mask]
        np.save(depth_path, depth_file)

        try:
            depth_vis, _ = visualize_depth_numpy(final_depth_values)
            depth_on_img = image.copy()
            # 簡單疊加
            mask_indices = np.where(mask)
            depth_on_img[mask_indices] = (0.5 * depth_on_img[mask_indices] + 0.5 * depth_vis[mask_indices]).astype(np.uint8)
            cv2.imwrite(depth_vis_path, depth_on_img)
        except Exception as e:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str, help='Path to the processed scene folder (e.g., data/sg_nuscenes/0061)')
    args = parser.parse_args()
    
    generate_lidar_depth(args.datadir)