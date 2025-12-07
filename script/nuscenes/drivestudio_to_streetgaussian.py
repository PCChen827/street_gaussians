import os
import argparse
import json
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from pyquaternion import Quaternion

# Street Gaussians default camera order
# DriveStudio order: 0:FRONT, 1:FRONT_LEFT, 2:FRONT_RIGHT, 3:BACK_LEFT, 4:BACK_RIGHT, 5:BACK
# Keep the same order
NUM_CAMS = 6

def parse_intrinsics(ds_intrinsic_file):
    """
    DriveStudio stores intrinsics as a flattened list of 9 elements:
    [fx, fy, cx, cy, 0, 0, 0, 0, 0]
    """
    data = np.loadtxt(ds_intrinsic_file)
    K = np.eye(3)
    K[0, 0] = data[0] # fx
    K[1, 1] = data[1] # fy
    K[0, 2] = data[2] # cx
    K[1, 2] = data[3] # cy
    return K

def convert_scene(src_scene_dir, dst_scene_dir):
    if os.path.exists(dst_scene_dir):
        print(f"Warning: {dst_scene_dir} exists. Overwriting...")
        # shutil.rmtree(dst_scene_dir) 
    
    os.makedirs(dst_scene_dir, exist_ok=True)
    
    # Build the dir
    subdirs = ['images', 'ego_pose', 'intrinsics', 'extrinsics', 'track', 'dynamic_mask', 'lidar']
    for d in subdirs:
        os.makedirs(os.path.join(dst_scene_dir, d), exist_ok=True)

    # Get the frames
    src_images = sorted([f for f in os.listdir(os.path.join(src_scene_dir, 'images')) if f.endswith('.jpg')])
    # DriveStudio naming: {frame_idx:03d}_{cam_idx}.jpg
    # Get max frame index
    max_frame = 0
    for img in src_images:
        idx = int(img.split('_')[0])
        if idx > max_frame:
            max_frame = idx
    num_frames = max_frame + 1
    print(f"Found {num_frames} frames in {src_scene_dir}")

    # Process Ego Pose & Extrinsics
    # Use relationship from the Frame 0 as Extrinsics
    # DriveStudio: lidar_pose = Ego -> World
    # DriveStudio: extrinsics = Cam -> World (per frame)
    
    # Read Ego Pose of Frame 0 (Lidar Pose)
    lidar_pose_0_path = os.path.join(src_scene_dir, 'lidar_pose', '000.txt')
    ego_to_world_0 = np.loadtxt(lidar_pose_0_path)
    world_to_ego_0 = np.linalg.inv(ego_to_world_0)

    # Process every camera's Extrinsics (Cam -> Ego) and Intrinsics
    for cam_id in range(NUM_CAMS):
        # --- Extrinsics ---
        # Read Frame 0 Cam Pose
        cam_pose_0_path = os.path.join(src_scene_dir, 'extrinsics', f'000_{cam_id}.txt')
        if os.path.exists(cam_pose_0_path):
            cam_to_world_0 = np.loadtxt(cam_pose_0_path)
            # Cam -> Ego = (World -> Ego) @ (Cam -> World)
            cam_to_ego = world_to_ego_0 @ cam_to_world_0
            np.savetxt(os.path.join(dst_scene_dir, 'extrinsics', f'{cam_id}.txt'), cam_to_ego)
        
        # --- Intrinsics ---
        int_path = os.path.join(src_scene_dir, 'intrinsics', f'{cam_id}.txt')
        if os.path.exists(int_path):
            K = parse_intrinsics(int_path)
            np.savetxt(os.path.join(dst_scene_dir, 'intrinsics', f'{cam_id}.txt'), K)

    # Process frame by frame: Images, Ego Pose, Dynamic Mask
    for frame_id in tqdm(range(num_frames), desc="Processing Frames"):
        frame_str_src = f"{frame_id:03d}"
        frame_str_dst = f"{frame_id:06d}"
        
        # --- Ego Pose ---
        src_pose = os.path.join(src_scene_dir, 'lidar_pose', f'{frame_str_src}.txt')
        dst_pose = os.path.join(dst_scene_dir, 'ego_pose', f'{frame_str_dst}.txt')
        shutil.copy2(src_pose, dst_pose)

        # --- Images & Masks ---
        for cam_id in range(NUM_CAMS):
            # Image
            src_img = os.path.join(src_scene_dir, 'images', f'{frame_str_src}_{cam_id}.jpg')
            dst_img = os.path.join(dst_scene_dir, 'images', f'{frame_str_dst}_{cam_id}.png')
            
            if os.path.exists(src_img):
                # Save jpg with png
                img = cv2.imread(src_img)
                cv2.imwrite(dst_img, img)

            # Dynamic Mask (DriveStudio: dynamic_masks/all/)
            src_mask = os.path.join(src_scene_dir, 'dynamic_masks', 'all', f'{frame_str_src}_{cam_id}.png')
            dst_mask = os.path.join(dst_scene_dir, 'dynamic_mask', f'{frame_str_dst}_{cam_id}.png')
            
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            else:
                # If there is no mask, generate whole black mask (no moving object)
                if os.path.exists(src_img):
                    cv2.imwrite(dst_mask, np.zeros(img.shape[:2], dtype=np.uint8))

    # Process Tracking Info
    # Read DriveStudio: instances_info.json
    # Structure: Dict[obj_id] -> { 'frame_annotations': { 'frame_idx': [], 'obj_to_world': [], 'box_size': [] } }
    json_path = os.path.join(src_scene_dir, 'instances', 'instances_info.json')
    track_file_path = os.path.join(dst_scene_dir, 'track', 'track_info.txt')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            instances = json.load(f)
            
        with open(track_file_path, 'w') as f_track:
            # Write Header
            f_track.write("frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed\n")
            
            for obj_id, data in instances.items():
                track_id = int(obj_id) # DriveStudio uses integer ID
                obj_class = data['class_name'].split('.')[1] if '.' in data['class_name'] else 'vehicle'
                
                # Revise name for Street Gaussians
                if 'vehicle' in data['class_name']: obj_class = 'vehicle'
                elif 'human' in data['class_name']: obj_class = 'pedestrian'
                
                anns = data['frame_annotations']
                for i, frame_idx in enumerate(anns['frame_idx']):
                    # Read Ego Pose of each frame (World -> Ego)
                    ego_pose_path = os.path.join(src_scene_dir, 'lidar_pose', f'{frame_idx:03d}.txt')
                    ego_to_world = np.loadtxt(ego_pose_path)
                    world_to_ego = np.linalg.inv(ego_to_world)
                    
                    # Object Pose (Obj -> World)
                    obj_to_world = np.array(anns['obj_to_world'][i])
                    
                    # Object in Ego Frame (Obj -> Ego = World -> Ego @ Obj -> World)
                    obj_to_ego = world_to_ego @ obj_to_world
                    
                    # Get position
                    tx, ty, tz = obj_to_ego[:3, 3]
                    
                    # Get rotation (Heading/Yaw)
                    # Assume Z axis point up
                    # rot matrix [0,0] = cos, [1,0] = sin
                    heading = np.arctan2(obj_to_ego[1, 0], obj_to_ego[0, 0])
                    
                    w, l, h = anns['box_size'][i]
                    
                    speed = 0.0
                    
                    line = f"{frame_idx} {track_id} {obj_class} -10 {h} {w} {l} {tx} {ty} {tz} {heading} {speed}\n"
                    f_track.write(line)

    # Process PointCloud (Combine all of the bin file to as npz file)
    pts_3d_all = {}
    pts_2d_all = {} # Dummy
    
    print("Generating pointcloud.npz ...")
    for frame_id in range(num_frames):
        bin_path = os.path.join(src_scene_dir, 'lidar', f'{frame_id:03d}.bin')
        if os.path.exists(bin_path):
            # Read float32 binary (x, y, z, intensity)
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            pts_3d = points[:, :3]
            
            pts_3d_all[frame_id] = pts_3d
            # Dummy projection: -1
            pts_2d_all[frame_id] = np.full((pts_3d.shape[0], 6), -1, dtype=np.int16)

    np.savez_compressed(
        os.path.join(dst_scene_dir, 'pointcloud.npz'),
        pointcloud=pts_3d_all,
        camera_projection=pts_2d_all
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_path', type=str, required=True, help='Path to DriveStudio processed scene folder (e.g. .../processed/v1.0-mini/061)')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save Street Gaussians format')
    args = parser.parse_args()

    convert_scene(args.ds_path, args.out_path)
    print(f"Done! Data saved to {args.out_path}")

if __name__ == '__main__':
    main()