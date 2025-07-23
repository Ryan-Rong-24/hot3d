# Hand Joints Extraction from HOT3D

This tool extracts 21 hand joint poses from HOT3D dataset sequences, converting from the existing format to detailed joint-level data with both position and rotation information.

## Overview

The script `extract_hand_joints.py` processes HOT3D sequences and extracts:
- **Hand Joint Data**: 21 joints per hand in 9D format (3D position + 6D rotation)
- **Object Poses**: Object trajectories in 6D rotation format
- **Actual Joint Rotations**: Uses MANO or UmeTrack hand models to compute real joint rotations

## Features

- **Real Joint Rotations**: Extracts actual joint rotations from hand models instead of identity matrices
- **Multiple Hand Models**: Supports both MANO and UmeTrack hand models
- **6D Rotation Format**: Uses 6D rotation representation for better neural network training
- **Batch Processing**: Processes multiple sequences efficiently
- **Error Handling**: Graceful handling of missing data and corrupted sequences

## Usage

### Basic Usage

```bash
# Navigate to the hot3d/hot3d directory
cd hot3d/hot3d

# Run with pixi (recommended)
pixi run python extract_hand_joints.py \
    --sequences_folder dataset \
    --object_library_folder dataset/assets

# Or activate the environment first
pixi shell
python extract_hand_joints.py \
    --sequences_folder dataset \
    --object_library_folder dataset/assets
```

### Advanced Usage

```bash
# Process with UmeTrack hand model
python extract_hand_joints.py \
    --sequences_folder dataset \
    --object_library_folder dataset/assets \
    --hand_type UMETRACK \
    --output ../../data/processed_hand_joints_umetrack.pkl

# Process limited sequences for testing
python extract_hand_joints.py \
    --sequences_folder dataset \
    --object_library_folder dataset/assets \
    --max_sequences 5 \
    --output ../../data/test_hand_joints.pkl
```

## Arguments

- `--sequences_folder`: Path to folder containing HOT3D sequences (default: `dataset`)
- `--object_library_folder`: Path to object library folder (default: `dataset/assets`)
- `--mano_model_folder`: Path to MANO models folder (optional)
- `--hand_type`: Hand model type - "MANO" (default) or "UMETRACK"
- `--output`: Output path for processed data (default: `../../data/processed_hand_joints.pkl`)
- `--max_sequences`: Maximum number of sequences to process (for testing)

## Output Format

The script generates a pickle file with the following structure:

```python
processed_data = {
    'sequence_name': {
        'left_hand': {
            'timestamps': np.array([...]),          # [T] timestamps
            'poses_9d': np.array([...])             # [T, 21, 9] joint poses
        },
        'right_hand': {
            'timestamps': np.array([...]),          # [T] timestamps  
            'poses_9d': np.array([...])             # [T, 21, 9] joint poses
        },
        'objects': {
            'object_uid': {
                'timestamps': np.array([...]),      # [T] timestamps
                'poses_9d': np.array([...])         # [T, 9] object poses
            }
        }
    }
}
```

### Joint Pose Format (9D)

Each joint pose contains:
- **Position** (3D): [x, y, z] coordinates in meters
- **Rotation** (6D): 6D rotation representation (more stable than quaternions)

The 6D rotation format encodes rotation matrices as the first two columns, which can be converted back to full rotation matrices for downstream tasks.

## Hand Models

### MANO (Default)
- 15 articulated joints with 3 DOF each (45 parameters)
- Converts axis-angle representations to rotation matrices
- Pads to 21 landmarks with identity rotations

### UmeTrack
- 16 articulated joints in kinematic chain
- Extracts joint rotations from rotation axes and joint angles
- Pads to 21 landmarks with identity rotations for fingertips

## Inspection Tool

Use the inspection script to analyze the generated data:

```bash
# Inspect generated hand joint data (run from hot3d/hot3d/)
python inspect_hand_joints.py ../../data/processed_hand_joints.pkl

# Compare with identity rotations
python inspect_hand_joints.py ../../data/processed_hand_joints.pkl --check_rotations
```

## Troubleshooting

### Import Errors
If you see import errors, make sure you're in the correct directory and using pixi:

```bash
# Navigate to the correct directory
cd hot3d/hot3d

# Use pixi environment
pixi shell

# Check that you can import the modules
python -c "from dataset_api import Hot3dDataProvider; print('Imports working!')"
```

### Missing Dependencies
The script requires:
- PyTorch and PyTorch3D
- NumPy and SciPy
- HOT3D dataset API
- projectaria_tools

Install with pixi:
```bash
cd hot3d/hot3d
pixi install
```

### Memory Issues
For large datasets, you can:
1. Use `--max_sequences` to process in batches
2. Process sequences individually
3. Increase system memory or use a machine with more RAM

### No Hand Data
If sequences have no hand data:
1. Check if the sequence contains hand tracking data
2. Verify the hand model type matches the sequence data
3. Check the `fail_on_missing_data=False` parameter is set

## Data Format Compatibility

The output format is compatible with existing preprocessing pipelines that expect:
- Timestamps as numpy arrays
- Pose data in [T, joints, 9] format
- Object trajectories in 6D rotation format

## Performance Notes

- Processing time depends on sequence length and number of sequences
- UmeTrack is generally faster than MANO for joint rotation computation
- Large sequences (>10k frames) may take several minutes each
- Consider using `--max_sequences` for initial testing

## Examples

See the generated data structure:
```python
import pickle
import numpy as np

# Load processed data (adjust path as needed)
with open('../../data/processed_hand_joints.pkl', 'rb') as f:
    data = pickle.load(f)

# Check first sequence
demo_id = list(data.keys())[0]
demo_data = data[demo_id]

print(f"Demo: {demo_id}")
if 'left_hand' in demo_data:
    print(f"Left hand: {demo_data['left_hand']['poses_9d'].shape}")
if 'right_hand' in demo_data:
    print(f"Right hand: {demo_data['right_hand']['poses_9d'].shape}")
``` 