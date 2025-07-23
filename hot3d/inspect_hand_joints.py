#!/usr/bin/env python3
import pickle
import argparse
import numpy as np
import sys

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def analyze_rotation_data(rotation_6d):
    """Analyze 6D rotation data to check if it's identity or actual rotations."""
    # Identity rotation in 6D representation is [1, 0, 0, 0, 1, 0]
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # Check if all rotations are close to identity
    is_identity = np.allclose(rotation_6d, identity_6d, atol=1e-6)
    
    # Compute variance to see how much the rotations vary
    variance = np.var(rotation_6d, axis=0)
    mean_variance = np.mean(variance)
    
    return is_identity, mean_variance

def inspect_hand_joints_data(data_path):
    """Inspect the processed hand joints data and show structure."""
    print(f"Loading processed hand joints data from {data_path}...")
    processed_data = load_pickle(data_path)
    
    print(f"\nFound {len(processed_data)} demonstrations:")
    
    for demo_id, demo_data in processed_data.items():
        print(f"\nDemo ID: {demo_id}")
        
        # Left hand info
        if 'left_hand' in demo_data and 'poses_9d' in demo_data['left_hand']:
            left_shape = demo_data['left_hand']['poses_9d'].shape
            if len(left_shape) == 3:  # [T, joints, 9]
                print(f"  Left hand: {left_shape} (T={left_shape[0]}, joints={left_shape[1]}, D={left_shape[2]})")
                
                # Show sample joint data
                sample_frame = demo_data['left_hand']['poses_9d'][0]  # First frame
                print(f"    Sample joint 0: pos={sample_frame[0][:3]}, rot_6d={sample_frame[0][3:]}")
                print(f"    Sample joint 5: pos={sample_frame[5][:3]}, rot_6d={sample_frame[5][3:]}")
                
                # Analyze rotation data
                all_rotations = demo_data['left_hand']['poses_9d'][:, :, 3:]  # [T, 21, 6]
                flat_rotations = all_rotations.reshape(-1, 6)  # [T*21, 6]
                is_identity, variance = analyze_rotation_data(flat_rotations)
                
                if is_identity:
                    print(f"    ⚠️  Rotations appear to be identity (variance: {variance:.2e})")
                else:
                    print(f"    ✓ Rotations contain actual joint rotations (variance: {variance:.2e})")
            else:
                print(f"  Left hand: {left_shape} (unexpected shape)")
        
        # Right hand info  
        if 'right_hand' in demo_data and 'poses_9d' in demo_data['right_hand']:
            right_shape = demo_data['right_hand']['poses_9d'].shape
            if len(right_shape) == 3:  # [T, joints, 9]
                print(f"  Right hand: {right_shape} (T={right_shape[0]}, joints={right_shape[1]}, D={right_shape[2]})")
                
                # Show sample joint data
                sample_frame = demo_data['right_hand']['poses_9d'][0]  # First frame
                print(f"    Sample joint 0: pos={sample_frame[0][:3]}, rot_6d={sample_frame[0][3:]}")
                print(f"    Sample joint 5: pos={sample_frame[5][:3]}, rot_6d={sample_frame[5][3:]}")
                
                # Analyze rotation data
                all_rotations = demo_data['right_hand']['poses_9d'][:, :, 3:]  # [T, 21, 6]
                flat_rotations = all_rotations.reshape(-1, 6)  # [T*21, 6]
                is_identity, variance = analyze_rotation_data(flat_rotations)
                
                if is_identity:
                    print(f"    ⚠️  Rotations appear to be identity (variance: {variance:.2e})")
                else:
                    print(f"    ✓ Rotations contain actual joint rotations (variance: {variance:.2e})")
            else:
                print(f"  Right hand: {right_shape} (unexpected shape)")
        
        # Object info
        if 'objects' in demo_data:
            objects = demo_data['objects']
            print(f"  Objects ({len(objects)} total):")
            for obj_id, obj_data in objects.items():
                obj_shape = obj_data['poses_9d'].shape
                print(f"    {obj_id}: {obj_shape} (T={obj_shape[0]}, D={obj_shape[1]})")
        
        # Show only first demo in detail to avoid too much output
        if list(processed_data.keys()).index(demo_id) >= 2:
            print(f"\n... and {len(processed_data) - 3} more demonstrations")
            break
    
    # Show expected format comparison
    print(f"\n" + "="*70)
    print("FORMAT COMPARISON:")
    print("="*70)
    
    first_demo_id = list(processed_data.keys())[0]
    first_demo = processed_data[first_demo_id]
    
    if 'left_hand' in first_demo:
        left_shape = first_demo['left_hand']['poses_9d'].shape
        print(f"NEW FORMAT - Left hand: {left_shape}")
        print(f"  - Time steps: {left_shape[0]}")
        print(f"  - Number of joints: {left_shape[1]} (expected: 21)")
        print(f"  - Data dimensions: {left_shape[2]} (expected: 9 = 3 position + 6 rotation)")
        
        if left_shape[1] == 21 and left_shape[2] == 9:
            print("  ✓ Format matches expected shape [T, 21, 9]")
        else:
            print("  ✗ Format does not match expected shape [T, 21, 9]")
        
        # Check rotation quality across all demos
        print(f"\n  ROTATION DATA ANALYSIS:")
        identity_count = 0
        actual_count = 0
        
        for demo_id, demo_data in processed_data.items():
            for hand_type in ['left_hand', 'right_hand']:
                if hand_type in demo_data and 'poses_9d' in demo_data[hand_type]:
                    rotations = demo_data[hand_type]['poses_9d'][:, :, 3:]
                    flat_rotations = rotations.reshape(-1, 6)
                    is_identity, _ = analyze_rotation_data(flat_rotations)
                    
                    if is_identity:
                        identity_count += 1
                    else:
                        actual_count += 1
        
        total_hands = identity_count + actual_count
        if total_hands > 0:
            print(f"  - Hands with actual joint rotations: {actual_count}/{total_hands} ({100*actual_count/total_hands:.1f}%)")
            print(f"  - Hands with identity rotations: {identity_count}/{total_hands} ({100*identity_count/total_hands:.1f}%)")
            
            if actual_count > 0:
                print("  ✓ Successfully extracted actual joint rotations!")
            else:
                print("  ⚠️  All rotations are identity - check hand model extraction")
    
    print(f"\nFor comparison, old format was [T, 9] where each hand was a single pose.")
    print(f"New format is [T, 21, 9] where each hand has 21 joint poses with actual rotations.")
    
    # Hand joint indices reference
    print(f"\n" + "="*50)
    print("HAND JOINT INDICES (21 landmarks):")
    print("="*50)
    joint_names = [
        "0: Thumb fingertip", "1: Index fingertip", "2: Middle fingertip", 
        "3: Ring fingertip", "4: Pinky fingertip", "5: Wrist joint",
        "6: Thumb intermediate", "7: Thumb distal", "8: Index proximal",
        "9: Index intermediate", "10: Index distal", "11: Middle proximal",
        "12: Middle intermediate", "13: Middle distal", "14: Ring proximal",
        "15: Ring intermediate", "16: Ring distal", "17: Pinky proximal",
        "18: Pinky intermediate", "19: Pinky distal", "20: Palm center"
    ]
    
    for name in joint_names:
        print(f"  {name}")
    
    print(f"\n" + "="*50)
    print("ROTATION REPRESENTATION:")
    print("="*50)
    print("Each joint has 6D rotation representation (continuous, unique):")
    print("  - 6D rotation is more stable than quaternions for neural networks")
    print("  - Identity rotation: [1, 0, 0, 0, 1, 0]")
    print("  - Non-identity rotations indicate actual joint articulation")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect processed hand joints data and show structure."
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="data/processed_hand_joints.pkl",
        help="Path to processed hand joints pickle file"
    )
    
    args = parser.parse_args()
    
    try:
        inspect_hand_joints_data(args.data_path)
    except FileNotFoundError:
        print(f"Error: File not found: {args.data_path}")
        print("Make sure to run extract_hand_joints.py first to generate the data.")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main() 