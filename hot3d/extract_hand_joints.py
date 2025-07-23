#!/usr/bin/env python3
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
import os
import sys
from tqdm import tqdm

try:
    from dataset_api import Hot3dDataProvider
    from data_loaders.loader_object_library import load_object_library
    from data_loaders.mano_layer import loadManoHandModel
    from data_loaders.loader_hand_poses import HandType, Handedness
    from data_loaders.hand_common import LANDMARK_INDEX_TO_NAMING
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from quaternion import quat_to_6v
except ImportError as e:
    print(f"Error importing HOT3D modules: {e}")
    print("Please ensure the hot3d directory is properly set up.")
    sys.exit(1)


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    Args:
        axis_angle: tensor of shape [..., 3] - axis-angle representation
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    # Handle batch dimensions
    batch_shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.view(-1, 3)
    
    # Compute angle (magnitude of axis-angle vector)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    
    # Handle case where angle is zero (identity rotation)
    small_angle = angle < 1e-8
    angle = torch.where(small_angle, torch.ones_like(angle), angle)
    
    # Normalize axis
    axis = axis_angle / angle
    
    # Rodrigues' rotation formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Cross product matrix [axis]_x
    axis_x = axis[:, 0]
    axis_y = axis[:, 1] 
    axis_z = axis[:, 2]
    
    zeros = torch.zeros_like(axis_x)
    
    K = torch.stack([
        torch.stack([zeros, -axis_z, axis_y], dim=-1),
        torch.stack([axis_z, zeros, -axis_x], dim=-1),
        torch.stack([-axis_y, axis_x, zeros], dim=-1)
    ], dim=-2)
    
    # Identity matrix
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(axis_angle.shape[0], -1, -1)
    
    # Rodrigues' formula: R = I + sin(θ) * K + (1 - cos(θ)) * K^2
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)
    
    # Handle small angles (use identity matrix)
    small_angle_expanded = small_angle.unsqueeze(-1).expand(-1, 3, 3)
    R = torch.where(small_angle_expanded, I, R)
    
    return R.view(*batch_shape, 3, 3)


def matrix_to_rotation_6d(matrix):
    """
    Convert rotation matrix to 6D rotation representation.
    Args:
        matrix: tensor of shape [..., 3, 3] - rotation matrices
    Returns:
        6D rotation representation of shape [..., 6] (first two columns of rotation matrix)
    """
    # Extract first two columns of rotation matrix
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def compute_joint_rotations_umetrack(hand_data_provider, hand_pose_data):
    """
    Compute individual joint rotations for UmeTrack hand model.
    Returns: numpy array of shape [21, 6] - 6D rotation representations
    """
    if not hasattr(hand_data_provider, '_hand_model') or hand_data_provider._hand_model is None:
        return None
    
    try:
        hand_model = hand_data_provider._hand_model
        joint_angles = torch.tensor(hand_pose_data.joint_angles, dtype=torch.float64)
        
        # Get hand model parameters
        joint_rotation_axes = hand_model.joint_rotation_axes.double()
        joint_rest_positions = hand_model.joint_rest_positions.double()
        joint_parent = hand_model.joint_parent
        
        # Number of joints in the kinematic chain (16 for UmeTrack)
        num_joints = len(joint_angles)
        
        # Initialize joint transformations
        joint_rotations = []
        
        # Root transformation (wrist)
        wrist_pose_matrix = hand_pose_data.wrist_pose.to_matrix()
        wrist_pose_tensor = torch.from_numpy(wrist_pose_matrix).double()
        
        # Flip for right hand
        if hand_pose_data.handedness == Handedness.Right:
            wrist_pose_tensor[:, 0] *= -1
        
        # Compute individual joint rotations from joint angles
        for i in range(num_joints):
            if i < len(joint_rotation_axes):
                # Get rotation axis and angle
                rotation_axis = joint_rotation_axes[i]
                angle = joint_angles[i] if i < len(joint_angles) else 0.0
                
                # Create rotation matrix from axis-angle
                axis_angle = rotation_axis * angle
                rotation_matrix = axis_angle_to_matrix(axis_angle.unsqueeze(0)).squeeze(0)
                
                # Convert to 6D rotation representation
                rotation_6d = matrix_to_rotation_6d(rotation_matrix.unsqueeze(0)).squeeze(0)
                joint_rotations.append(rotation_6d.numpy())
            else:
                # Identity rotation for missing joints
                identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                joint_rotations.append(identity_6d.numpy())
        
        # Pad to 21 joints (landmarks) with identity rotations
        while len(joint_rotations) < 21:
            identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            joint_rotations.append(identity_6d)
        
        return np.stack(joint_rotations[:21])  # [21, 6]
        
    except Exception as e:
        print(f"Warning: Could not compute UmeTrack joint rotations: {e}")
        return None


def compute_joint_rotations_mano(hand_data_provider, hand_pose_data):
    """
    Compute individual joint rotations for MANO hand model.
    Returns: numpy array of shape [21, 6] - 6D rotation representations
    """
    if not hasattr(hand_data_provider, 'mano_layer') or hand_data_provider.mano_layer is None:
        return None
    
    try:
        # Get MANO joint angles (15 joints * 3 DOF = 45 parameters)
        joint_angles = np.array(hand_pose_data.joint_angles)
        
        if len(joint_angles) != 45:  # MANO has 45 joint angle parameters
            print(f"Warning: Expected 45 MANO joint angles, got {len(joint_angles)}")
            return None
        
        # Reshape to [15, 3] (15 joints, 3 DOF each)
        joint_angles = joint_angles.reshape(15, 3)
        joint_angles_tensor = torch.from_numpy(joint_angles).float()
        
        # Convert axis-angle to rotation matrices
        joint_rotations = []
        
        for i in range(15):
            axis_angle = joint_angles_tensor[i]  # [3]
            
            # Convert axis-angle to rotation matrix
            rotation_matrix = axis_angle_to_matrix(axis_angle.unsqueeze(0)).squeeze(0)  # [3, 3]
            
            # Convert to 6D rotation representation
            rotation_6d = matrix_to_rotation_6d(rotation_matrix.unsqueeze(0)).squeeze(0)  # [6]
            joint_rotations.append(rotation_6d.numpy())
        
        # MANO has 15 articulated joints, but we need 21 landmarks
        # Add identity rotations for the additional landmark points
        for i in range(6):  # Add 6 more to reach 21
            identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            joint_rotations.append(identity_6d)
        
        return np.stack(joint_rotations)  # [21, 6]
        
    except Exception as e:
        print(f"Warning: Could not compute MANO joint rotations: {e}")
        return None


def convert_landmarks_to_9d_with_rotations(landmarks, joint_rotations):
    """
    Convert hand landmarks (21 x 3) and joint rotations (21 x 6) to 9D format (21 x 9).
    landmarks: torch.Tensor of shape [21, 3] - 3D positions
    joint_rotations: numpy array of shape [21, 6] - 6D rotation representations
    Returns: numpy array of shape [21, 9] - [3 translation + 6 rotation_6d]
    """
    if landmarks is None:
        return None
    
    # Extract 3D positions
    positions = landmarks.numpy()  # [21, 3]
    
    if joint_rotations is not None and joint_rotations.shape == (21, 6):
        # Use actual joint rotations
        rotations_6d = joint_rotations  # [21, 6]
    else:
        # Fallback to identity rotations
        print("Warning: Using identity rotations for joints")
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        identity_quat = identity_quat.unsqueeze(0).repeat(21, 1)  # [21, 4]
        rotations_6d = quat_to_6v(identity_quat).numpy()  # [21, 6]
    
    # Combine translation and rotation
    poses_9d = np.concatenate([positions, rotations_6d], axis=-1)  # [21, 9]
    
    return poses_9d


def extract_hand_joints_from_sequence(sequence_folder, object_library_folder, 
                                     mano_model_folder=None, hand_type="UMETRACK"):
    """
    Extract hand joint data from a HOT3D sequence.
    Returns: dict with 'left_hand', 'right_hand', 'objects' data
    """
    print(f"Processing sequence: {sequence_folder}")
    
    # Validate paths
    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(object_library_folder):
        raise RuntimeError(f"Object library folder {object_library_folder} does not exist")
    
    # Load object library
    object_library = load_object_library(object_library_folderpath=object_library_folder)
    
    # Load MANO model if provided
    mano_hand_model = loadManoHandModel(mano_model_folder) if mano_model_folder else None
    
    # Initialize data provider
    hand_enum_type = HandType.Umetrack if hand_type == "UMETRACK" else HandType.Mano
    data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=False,
    )
    
    print(f"Data provider statistics: {data_provider.get_data_statistics()}")
    
    # Get hand data provider
    hand_data_provider = (data_provider.mano_hand_data_provider 
                         if data_provider.mano_hand_data_provider is not None 
                         else data_provider.umetrack_hand_data_provider)
    
    if hand_data_provider is None:
        raise RuntimeError("No hand data provider available")
    
    # Determine hand model type for rotation computation
    is_mano = data_provider.mano_hand_data_provider is not None
    print(f"Using {'MANO' if is_mano else 'UmeTrack'} hand model for joint rotations")
    
    # Get timestamps
    timestamps = data_provider.device_data_provider.get_sequence_timestamps()
    print(f"Found {len(timestamps)} timestamps")
    
    # Initialize storage
    left_hand_data = []
    right_hand_data = []
    object_data = []
    
    # Process each timestamp
    for timestamp_ns in tqdm(timestamps, desc="Extracting hand joints with rotations"):
        # Get hand poses
        hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        
        if hand_poses_with_dt is not None:
            hand_pose_collection = hand_poses_with_dt.pose3d_collection
            
            # Process each hand
            for hand_pose_data in hand_pose_collection.poses.values():
                # Get hand landmarks (positions)
                hand_landmarks = hand_data_provider.get_hand_landmarks(hand_pose_data)
                
                if hand_landmarks is not None:
                    # Compute joint rotations based on hand model type
                    if is_mano:
                        joint_rotations = compute_joint_rotations_mano(hand_data_provider, hand_pose_data)
                    else:
                        joint_rotations = compute_joint_rotations_umetrack(hand_data_provider, hand_pose_data)
                    
                    # Convert to 9D format with actual rotations
                    joints_9d = convert_landmarks_to_9d_with_rotations(hand_landmarks, joint_rotations)
                    
                    # Store data
                    frame_data = {
                        'timestamp_ns': timestamp_ns,
                        'joints_9d': joints_9d  # [21, 9]
                    }
                    
                    if hand_pose_data.is_left_hand():
                        left_hand_data.append(frame_data)
                    elif hand_pose_data.is_right_hand():
                        right_hand_data.append(frame_data)
        
        # Get object poses (keeping existing object extraction logic)
        object_poses_with_dt = None
        if data_provider.object_pose_data_provider is not None:
            object_poses_with_dt = data_provider.object_pose_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
        
        if object_poses_with_dt is not None:
            objects_pose3d_collection = object_poses_with_dt.pose3d_collection
            frame_objects = []
            
            for object_uid, object_pose3d in objects_pose3d_collection.poses.items():
                # Convert SE3 to translation + quaternion
                if object_pose3d is None or object_pose3d.T_world_object is None:
                    continue
                    
                translation = object_pose3d.T_world_object.translation()
                
                # Get rotation - handle different SO3 object types
                rotation_quat = [1.0, 0.0, 0.0, 0.0]  # Default identity quaternion
                rotation_matrix = None
                
                try:
                    rotation_so3 = object_pose3d.T_world_object.rotation()
                    
                    # Try to get quaternion directly from axis-angle representation
                    if hasattr(rotation_so3, 'log'):
                        # Convert from axis-angle to quaternion
                        axis_angle = rotation_so3.log()
                        from scipy.spatial.transform import Rotation
                        rotation_quat_raw = Rotation.from_rotvec(axis_angle).as_quat(canonical=True)  # [x, y, z, w]
                        
                        # Handle both 1D and 2D quaternion arrays
                        if rotation_quat_raw.ndim == 2 and rotation_quat_raw.shape[0] == 1:
                            # Shape is (1, 4), flatten it
                            rotation_quat_raw = rotation_quat_raw.flatten()
                        
                        # Check if we got a valid quaternion array
                        if len(rotation_quat_raw) == 4:
                            rotation_quat = [rotation_quat_raw[3], rotation_quat_raw[0], rotation_quat_raw[1], rotation_quat_raw[2]]  # [w, x, y, z]
                        else:
                            print(f"Warning: Unexpected quaternion format from axis-angle conversion, got shape {rotation_quat_raw.shape}")
                            rotation_quat = [1.0, 0.0, 0.0, 0.0]
                    else:
                        # Try different methods to get rotation matrix
                        if hasattr(rotation_so3, 'matrix'):
                            rotation_matrix = rotation_so3.matrix()
                        elif hasattr(rotation_so3, 'to_matrix'):
                            rotation_matrix = rotation_so3.to_matrix()
                        elif hasattr(rotation_so3, 'as_matrix'):
                            rotation_matrix = rotation_so3.as_matrix()
                        
                        # Convert rotation matrix to quaternion if we got one
                        if rotation_matrix is not None:
                            from scipy.spatial.transform import Rotation
                            rotation_quat_raw = Rotation.from_matrix(rotation_matrix).as_quat(canonical=True)  # [x, y, z, w]
                            
                            # Handle both 1D and 2D quaternion arrays
                            if rotation_quat_raw.ndim == 2 and rotation_quat_raw.shape[0] == 1:
                                # Shape is (1, 4), flatten it
                                rotation_quat_raw = rotation_quat_raw.flatten()
                            
                            # Check if we got a valid quaternion array
                            if len(rotation_quat_raw) == 4:
                                rotation_quat = [rotation_quat_raw[3], rotation_quat_raw[0], rotation_quat_raw[1], rotation_quat_raw[2]]  # [w, x, y, z]
                            else:
                                print(f"Warning: Unexpected quaternion format from matrix conversion, got shape {rotation_quat_raw.shape}")
                                rotation_quat = [1.0, 0.0, 0.0, 0.0]
                        else:
                            print(f"Warning: Could not extract rotation from SO3 object, using identity quaternion")
                            
                except Exception as e:
                    print(f"Warning: Could not extract rotation from SO3 object: {e}, using identity quaternion")
                    rotation_quat = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
                
                frame_objects.append({
                    'object_uid': object_uid,
                    'translation': translation,
                    'rotation': [rotation_quat]  # Nested list for compatibility
                })
            
            object_data.append({
                'timestamp_ns': timestamp_ns,
                'poses': frame_objects
            })
    
    return {
        'left_hand': left_hand_data,
        'right_hand': right_hand_data,
        'object_pose': object_data
    }


def convert_hand_joints_to_processed_format(hand_data):
    """
    Convert hand joint data to the processed format.
    hand_data: list of frames with 'timestamp_ns' and 'joints_9d'
    Returns: dict with 'timestamps', 'poses_9d' where poses_9d is [T, 21, 9]
    """
    if not hand_data:
        return {'timestamps': np.array([]), 'poses_9d': np.array([])}
    
    timestamps = []
    joints_9d_list = []
    
    for frame in hand_data:
        timestamps.append(frame['timestamp_ns'])
        joints_9d_list.append(frame['joints_9d'])  # [21, 9]
    
    timestamps = np.array(timestamps)
    joints_9d = np.stack(joints_9d_list, axis=0)  # [T, 21, 9]
    
    return {
        'timestamps': timestamps,
        'poses_9d': joints_9d
    }


def convert_object_trajectory_to_6d(object_data):
    """
    Convert object trajectory data from quaternion to 6D rotation format.
    (Reusing the existing logic from preprocess_data.py)
    """
    if not object_data:
        return {}
    
    # Group by object_uid
    object_trajectories = {}
    
    for frame in object_data:
        timestamp = frame['timestamp_ns']
        for obj in frame['poses']:
            obj_uid = obj['object_uid']
            if obj_uid not in object_trajectories:
                object_trajectories[obj_uid] = {
                    'timestamps': [],
                    'translations': [],
                    'rotations': []
                }
            
            object_trajectories[obj_uid]['timestamps'].append(timestamp)
            object_trajectories[obj_uid]['translations'].append(obj['translation'])
            object_trajectories[obj_uid]['rotations'].append(obj['rotation'][0])
    
    # Convert each object trajectory to 6D format
    processed_objects = {}
    for obj_uid, obj_data in object_trajectories.items():
        timestamps = np.array(obj_data['timestamps'])
        translations = torch.tensor(obj_data['translations'], dtype=torch.float32)
        rotations = torch.tensor(obj_data['rotations'], dtype=torch.float32)
        
        # Convert quaternions to 6D rotation
        rotations_6d = quat_to_6v(rotations)
        
        # Combine translation and rotation
        poses_9d = torch.cat([translations, rotations_6d], dim=-1)
        
        processed_objects[obj_uid] = {
            'timestamps': timestamps,
            'poses_9d': poses_9d.numpy()
        }
    
    return processed_objects


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand joint data from HOT3D sequences and save in processed format."
    )
    parser.add_argument(
        "--sequences_folder",
        type=str,
        required=True,
        help="Path to folder containing HOT3D sequences"
    )
    parser.add_argument(
        "--object_library_folder",
        type=str,
        required=True,
        help="Path to object library folder containing instance.json and *.glb files"
    )
    parser.add_argument(
        "--mano_model_folder",
        type=str,
        default=None,
        help="Path to MANO models folder (optional)"
    )
    parser.add_argument(
        "--hand_type",
        type=str,
        default="MANO",
        choices=["UMETRACK", "MANO"],
        help="Type of hand model to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../data/processed_hand_joints.pkl",
        help="Output path for processed data"
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.sequences_folder):
        print(f"Error: Sequences folder {args.sequences_folder} not found!")
        return
    
    if not os.path.exists(args.object_library_folder):
        print(f"Error: Object library folder {args.object_library_folder} not found!")
        return
    
    # Find all sequence folders
    sequence_folders = []
    for item in os.listdir(args.sequences_folder):
        item_path = os.path.join(args.sequences_folder, item)
        # Skip assets folder and only process directories that look like sequences
        if (os.path.isdir(item_path) and 
            item != 'assets' and 
            not item.startswith('.')):
            sequence_folders.append(item_path)
    
    print(f"Found {len(sequence_folders)} sequences")
    
    if args.max_sequences:
        sequence_folders = sequence_folders[:args.max_sequences]
        print(f"Processing first {len(sequence_folders)} sequences")
    
    # Process each sequence
    processed_data = {}
    
    for sequence_folder in sequence_folders:
        sequence_name = os.path.basename(sequence_folder)
        print(f"\n{'='*50}")
        print(f"Processing sequence: {sequence_name}")
        print(f"{'='*50}")
        
        try:
            # Extract data from sequence
            raw_data = extract_hand_joints_from_sequence(
                sequence_folder=sequence_folder,
                object_library_folder=args.object_library_folder,
                mano_model_folder=args.mano_model_folder,
                hand_type=args.hand_type
            )
            
            # Convert to processed format
            processed_demo = {}
            
            # Process left hand
            if raw_data['left_hand']:
                print(f"  Left hand: {len(raw_data['left_hand'])} frames")
                processed_demo['left_hand'] = convert_hand_joints_to_processed_format(raw_data['left_hand'])
                print(f"  Left hand processed shape: {processed_demo['left_hand']['poses_9d'].shape}")
            
            # Process right hand
            if raw_data['right_hand']:
                print(f"  Right hand: {len(raw_data['right_hand'])} frames")
                processed_demo['right_hand'] = convert_hand_joints_to_processed_format(raw_data['right_hand'])
                print(f"  Right hand processed shape: {processed_demo['right_hand']['poses_9d'].shape}")
            
            # Process objects
            if raw_data['object_pose']:
                print(f"  Object poses: {len(raw_data['object_pose'])} frames")
                processed_demo['objects'] = convert_object_trajectory_to_6d(raw_data['object_pose'])
                print(f"  Found {len(processed_demo['objects'])} unique objects")
            
            processed_data[sequence_name] = processed_demo
            
        except Exception as e:
            print(f"Error processing sequence {sequence_name}: {e}")
            continue
    
    # Save processed data
    print(f"\nSaving processed data to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Processing completed!")
    
    # Print summary
    print(f"\nSummary: Processed {len(processed_data)} sequences")
    for demo_id, demo_data in processed_data.items():
        print(f"\nDemo {demo_id}:")
        if 'left_hand' in demo_data:
            shape = demo_data['left_hand']['poses_9d'].shape
            print(f"  Left hand: {shape} (T={shape[0]}, joints={shape[1]}, D={shape[2]})")
        if 'right_hand' in demo_data:
            shape = demo_data['right_hand']['poses_9d'].shape
            print(f"  Right hand: {shape} (T={shape[0]}, joints={shape[1]}, D={shape[2]})")
        if 'objects' in demo_data:
            print(f"  Objects: {len(demo_data['objects'])} unique objects")
            for obj_id, obj_data in demo_data['objects'].items():
                print(f"    {obj_id}: {obj_data['poses_9d'].shape}")
        
        # Only show first demo details
        break


if __name__ == "__main__":
    main() 