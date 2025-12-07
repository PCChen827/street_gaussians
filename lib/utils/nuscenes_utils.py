import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm 
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import get_rays, sphere_intersection
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm

# Setting of 6 cameras
_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
    'SIDE_LEFT': 3,
    'SIDE_RIGHT': 4,
    'BACK': 5, 
}

# nuScenes resolution
image_heights = [900] * 6
image_widths = [1600] * 6

image_filename_to_cam = lambda x: int(x.split('.')[0].split('_')[-1])
image_filename_to_frame = lambda x: int(x.split('.')[0].split('_')[0])

def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    
    intrinsics = []
    extrinsics = []
    
    num_cams = len(glob(os.path.join(intrinsics_dir, "*.txt")))
    if num_cams == 0: num_cams = 6 

    for i in range(num_cams):
        int_path = os.path.join(intrinsics_dir,  f"{i}.txt")
        if not os.path.exists(int_path):
             intrinsics.append(np.eye(3))
             continue
        
        intrinsic = np.loadtxt(int_path)
        if intrinsic.ndim == 2 and intrinsic.shape == (3, 3):
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        else:
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        
    for i in range(num_cams):
        ext_path = os.path.join(extrinsics_dir,  f"{i}.txt")
        if os.path.exists(ext_path):
            cam_to_ego = np.loadtxt(ext_path)
            extrinsics.append(cam_to_ego)
        else:
            extrinsics.append(np.eye(4))
    
    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(num_cams)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    
    for ego_pose_path in ego_pose_paths:
        path = os.path.join(ego_pose_dir, ego_pose_path)
        if '_' not in ego_pose_path:
            ego_frame_poses.append(np.loadtxt(path))
        else:
            try:
                cam = image_filename_to_cam(ego_pose_path)
                if cam < num_cams:
                    ego_cam_poses[cam].append(np.loadtxt(path))
            except: pass
    
    ego_frame_poses = np.array(ego_frame_poses)
    if len(ego_frame_poses) > 0:
        center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
        ego_frame_poses[:, :3, 3] -= center_point 
        
        new_ego_cam_poses = []
        for i in range(num_cams):
            if len(ego_cam_poses[i]) > 0:
                cp = np.array(ego_cam_poses[i])
                cp[:, :3, 3] -= center_point
                new_ego_cam_poses.append(cp)
            else:
                new_ego_cam_poses.append(ego_frame_poses)
        ego_cam_poses = np.array(new_ego_cam_poses)
    else:
        ego_cam_poses = np.zeros((num_cams, 1, 4, 4))
        
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses

def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])
    return obj_pose_vehicle, obj_pose_world

def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4, 5]):
    tracklets_ls = []    
    objects_info = {}

    if cfg.data.get('use_tracker', False):
        tracklet_path = os.path.join(datadir, 'track/track_info_castrack.txt')
        vis_path = os.path.join(datadir, 'track/track_camera_vis_castrack.json')
    else:
        tracklet_path = os.path.join(datadir, 'track/track_info.txt')
        vis_path = os.path.join(datadir, 'track/track_camera_vis.json')

    if not os.path.exists(tracklet_path):
        return np.zeros((0, 0, 7)), np.zeros((0, 0, 7)), {}

    with open(tracklet_path, 'r') as f:
        tracklets_str = f.read().splitlines()[1:]
    
    has_vis = os.path.exists(vis_path)
    if has_vis:
        with open(vis_path, 'r') as f:
            tracklet_camera_vis = json.load(f)
    
    start_frame, end_frame = selected_frames[0], selected_frames[1]
    n_frames = len(ego_poses)
    n_obj_in_frame = np.zeros(n_frames)
    
    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id, track_id = int(tracklet[0]), int(tracklet[1])
        object_class = tracklet[2]
        if object_class in ['sign', 'misc']: continue
        
        if has_vis:
            try:
                vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
                if len(set(cameras) & set(vis_list)) == 0: continue
            except: pass
                
        if track_id not in objects_info:
            objects_info[track_id] = {
                'track_id': track_id, 'class': object_class, 'class_label': 0,
                'height': float(tracklet[4]), 'width': float(tracklet[5]), 'length': float(tracklet[6])
            }
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))
            
        t_arr = np.concatenate([np.array(tracklet[:2], float), [0], np.array(tracklet[4:], float)])
        tracklets_ls.append(t_arr)
        if frame_id < n_frames: n_obj_in_frame[frame_id] += 1
        
    tracklets_array = np.array(tracklets_ls)
    max_obj = int(n_obj_in_frame[start_frame:end_frame+1].max()) if len(tracklets_array) > 0 else 0
    num_frames_subset = end_frame - start_frame + 1
    
    vis_ids = np.full((num_frames_subset, max_obj), -1.0)
    vis_pose_veh = np.full((num_frames_subset, max_obj, 7), -1.0)
    vis_pose_wld = np.full((num_frames_subset, max_obj, 7), -1.0)
    
    for tr in tracklets_array:
        fid, tid = int(tr[0]), int(tr[1])
        if start_frame <= fid <= end_frame:
            ep = ego_poses[fid]
            pv, pw = make_obj_pose(ep, tr[6:10])
            f_idx = fid - start_frame
            slots = np.where(vis_ids[f_idx] < 0)[0]
            if len(slots) > 0:
                slot = slots[0]
                vis_ids[f_idx, slot] = tid
                vis_pose_veh[f_idx, slot] = pv
                vis_pose_wld[f_idx, slot] = pw

    box_scale = cfg.data.get('box_scale', 1.0)
    frames_range = np.arange(start_frame, end_frame + 1)
    
    # Filter out the object only in 1 (or 0) frame
    valid_objects = {}
    for tid, obj in objects_info.items():
        f_indices = np.where(vis_ids == tid)[0]
        
        if len(f_indices) < 2:
            # Remove the object: set vis_ids as -1
            vis_ids[vis_ids == tid] = -1.0
            continue

        obj['deformable'] = (obj['class'] == 'pedestrian')
        obj['width'] *= box_scale
        obj['length'] *= box_scale
        
        obj_frames = frames_range[f_indices]
        obj['start_frame'], obj['end_frame'] = obj_frames.min(), obj_frames.max()
        valid_objects[tid] = obj
            
    return np.dstack([vis_ids[...,None], vis_pose_wld]), np.dstack([vis_ids[...,None], vis_pose_veh]), valid_objects

def generate_dataparser_outputs(datadir, selected_frames=None, build_pointcloud=True, cameras=[0, 1, 2, 3, 4, 5]):
    image_dir = os.path.join(datadir, 'images')
    images = sorted(glob(os.path.join(image_dir, '*.png')))
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    n_cam_avail = len(glob(os.path.join(intrinsics_dir, "*.txt")))
    if n_cam_avail == 0: n_cam_avail = 6
    
    n_frames_total = len(images) // n_cam_avail
    if selected_frames is None:
        selected_frames = [0, n_frames_total - 1]
    start, end = selected_frames
    n_frames = end - start + 1
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)

    frames, frames_idx, cams, image_paths = [], [], [], []
    ixts, exts, poses, c2ws = [], [], [], []
    frame_ts, cam_ts = [], []
    
    ts_path = os.path.join(datadir, 'timestamps.json')
    if os.path.exists(ts_path):
        with open(ts_path) as f: timestamps = json.load(f)
    else:
        # print("Timestamps.json not found, generating dummy timestamps.")
        timestamps = {'FRAME': {}}
        for i in range(n_frames_total + 100):
            t = i * 0.1
            timestamps['FRAME'][f'{i:06d}'] = t
            for c in range(n_cam_avail):
                if str(c) not in timestamps: timestamps[str(c)] = {}
                timestamps[str(c)][f'{i:06d}'] = t

    for f in range(start, end + 1):
        frame_ts.append(timestamps['FRAME'][f'{f:06d}'])

    for img_path in images:
        base = os.path.basename(img_path)
        try:
            f, c = image_filename_to_frame(base), image_filename_to_cam(base)
        except: continue
        
        if start <= f <= end and c in cameras:
            if c >= len(intrinsics): continue
            ixt, ext = intrinsics[c], extrinsics[c]
            if c < len(ego_cam_poses) and len(ego_cam_poses[c]) > f:
                pose = ego_cam_poses[c][f]
            else:
                pose = ego_frame_poses[f] if f < len(ego_frame_poses) else np.eye(4)
            frames.append(f)
            frames_idx.append(f - start)
            cams.append(c)
            image_paths.append(img_path)
            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(pose @ ext)
            ts_key = str(c) if str(c) in timestamps else 'FRAME'
            cam_ts.append(timestamps[ts_key][f'{f:06d}'])

    exts = np.stack(exts)
    ixts = np.stack(ixts)
    poses = np.stack(poses)
    c2ws = np.stack(c2ws)
    
    ts_offset = min(cam_ts + frame_ts)
    cam_ts = np.array(cam_ts) - ts_offset
    frame_ts = np.array(frame_ts) - ts_offset
    
    _, obj_tracks, obj_info = get_obj_pose_tracking(datadir, selected_frames, ego_frame_poses, cameras)
    
    for tid, obj in obj_info.items():
        sf, ef = obj['start_frame'], obj['end_frame']
        t_s = timestamps['FRAME'].get(f'{sf:06d}', sf * 0.1)
        t_e = timestamps['FRAME'].get(f'{ef:06d}', ef * 0.1)
        obj['start_timestamp'] = t_s - ts_offset - 0.1
        obj['end_timestamp'] = t_e - ts_offset + 0.1

    result = {
        'num_frames': n_frames,
        'exts': exts, 'ixts': ixts, 'poses': poses, 'c2ws': c2ws,
        'obj_tracklets': obj_tracks, 'obj_info': obj_info,
        'frames': frames, 'cams': cams, 'frames_idx': frames_idx,
        'image_filenames': image_paths,
        'cams_timestamps': cam_ts, 'tracklet_timestamps': frame_ts
    }

    obj_bounds = []
    for i, path in tqdm(enumerate(image_paths), desc="Generating Obj Masks"):
        c = cams[i]
        h, w = (image_heights[c], image_widths[c]) if c < len(image_heights) else (900, 1600)
        mask = np.zeros((h, w), dtype=np.uint8)
        f_idx = frames_idx[i]
        if f_idx < len(obj_tracks):
            tracks = obj_tracks[f_idx]
            K, E = ixts[i], exts[i]
            for tr in tracks:
                tid = int(tr[0])
                if tid >= 0 and tid in obj_info:
                    op_v = np.eye(4)
                    op_v[:3, :3] = quaternion_to_matrix_numpy(tr[4:8])
                    op_v[:3, 3] = tr[1:4]
                    l, wd, ht = obj_info[tid]['length'], obj_info[tid]['width'], obj_info[tid]['height']
                    bbox = np.array([[-l, -wd, -ht], [l, wd, ht]]) * 0.5
                    corners = bbox_to_corner3d(bbox)
                    corners = np.concatenate([corners, np.ones_like(corners[..., :1])], axis=-1)
                    corners_v = corners @ op_v.T
                    m = get_bound_2d_mask(corners_v[..., :3], K, np.linalg.inv(E), h, w)
                    mask = np.logical_or(mask, m)
        obj_bounds.append(mask)
    result['obj_bounds'] = obj_bounds

    colmap_dir = os.path.join(cfg.model_path, 'colmap')
    if build_pointcloud:
        print('Building point cloud...')
        pc_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pc_dir, exist_ok=True)
        xyz_dict = {'bkgd': []}
        rgb_dict = {'bkgd': []}
        for tid in obj_info:
            xyz_dict[f'obj_{tid:03d}'] = []
            rgb_dict[f'obj_{tid:03d}'] = []
            
        pc_path = os.path.join(datadir, 'pointcloud.npz')
        if os.path.exists(pc_path):
            data = np.load(pc_path, allow_pickle=True)
            pts3d = data['pointcloud'].item()
            for f in tqdm(range(start, end + 1), desc="Processing LiDAR"):
                if f not in pts3d: continue
                raw_pts = pts3d[f]
                ep = ego_frame_poses[f] if f < len(ego_frame_poses) else np.eye(4)
                pts_h = np.concatenate([raw_pts, np.ones((len(raw_pts), 1))], axis=1)
                pts_w = pts_h @ ep.T
                pts_xyz_w = pts_w[:, :3]
                pts_rgb = np.full_like(pts_xyz_w, 0.5)
                
                is_obj = np.zeros(len(raw_pts), dtype=bool)
                if (f - start) < len(obj_tracks):
                    tracks = obj_tracks[f - start]
                    for tr in tracks:
                        tid = int(tr[0])
                        if tid >= 0 and tid in obj_info:
                            op_v = np.eye(4)
                            op_v[:3, :3] = quaternion_to_matrix_numpy(tr[4:8])
                            op_v[:3, 3] = tr[1:4]
                            v2l = np.linalg.inv(op_v)
                            pts_l = (np.concatenate([raw_pts, np.ones((len(raw_pts), 1))], axis=1) @ v2l.T)[:, :3]
                            l, wd, ht = obj_info[tid]['length'], obj_info[tid]['width'], obj_info[tid]['height']
                            bbox = [[-l/2, -wd/2, -ht/2], [l/2, wd/2, ht/2]]
                            in_box = inbbox_points(pts_l, bbox_to_corner3d(bbox))
                            is_obj |= in_box
                            xyz_dict[f'obj_{tid:03d}'].append(pts_l[in_box])
                            rgb_dict[f'obj_{tid:03d}'].append(pts_rgb[in_box])

                xyz_dict['bkgd'].append(pts_xyz_w[~is_obj])
                rgb_dict['bkgd'].append(pts_rgb[~is_obj])

        for k, v in xyz_dict.items():
            if len(v) == 0: continue
            xyz = np.concatenate(v, axis=0)
            if xyz.shape[0] == 0: continue # Check for empty array
            rgb = np.concatenate(rgb_dict[k], axis=0)
            if k == 'bkgd':
                try:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    pcd.colors = o3d.utility.Vector3dVector(rgb)
                    pcd = pcd.voxel_down_sample(voxel_size=0.2)
                    xyz = np.asarray(pcd.points)
                    rgb = np.asarray(pcd.colors)
                except: pass
            elif len(xyz) > 20000:
                idx = np.random.choice(len(xyz), 20000, replace=False)
                xyz, rgb = xyz[idx], rgb[idx]
            storePly(os.path.join(pc_dir, f'points3D_{k}.ply'), xyz, rgb)
            
        result['points_xyz_dict'] = {k: np.concatenate(v) if len(v)>0 else np.zeros((0,3)) for k,v in xyz_dict.items()}
        result['points_rgb_dict'] = {k: np.concatenate(v) if len(v)>0 else np.zeros((0,3)) for k,v in rgb_dict.items()}

    return result