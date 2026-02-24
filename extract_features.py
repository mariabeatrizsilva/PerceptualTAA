"""
Phase 2: Extract video features from 16xSSAA reference videos.

Extracts motion, complexity, and appearance features that may correlate with quality.
"""
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

SCENE_NAMES = [
    "abandoned", "abandoned-demo", "abandoned-flipped", "cubetest", 
    "fantasticvillage-open", "lightfoliage", "lightfoliage-close", 
    "oldmine", "oldmine-close", "oldmine-warm", "quarry-all", 
    "quarry-rocksonly", "resto-close", "resto-fwd", "resto-pan", 
    "scifi", "subway-lookdown", "subway-turn", "wildwest-bar", 
    "wildwest-barzoom", "wildwest-behindcounter", "wildwest-store", 
    "wildwest-town"
]


def compute_optical_flow_features(frame1, frame2):
    """Compute optical flow between two frames."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Calculate flow magnitude
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    return {
        'flow_mean': np.mean(magnitude),
        'flow_std': np.std(magnitude),
        'flow_max': np.max(magnitude),
        'flow_median': np.median(magnitude),
        'flow_95th': np.percentile(magnitude, 95)
    }


def compute_spatial_features(frame):
    """Compute spatial complexity features from a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Texture - using Laplacian variance as sharpness/detail measure
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = laplacian.var()
    
    # High frequency content (spatial frequency)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    
    # Get high frequency energy (outer regions of spectrum)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.uint8)
    r = 30  # Radius for low-pass (smaller = more high freq)
    cv2.circle(mask, (ccol, crow), r, 0, -1)
    high_freq_energy = np.sum(magnitude_spectrum * mask) / np.sum(magnitude_spectrum)
    
    # Contrast
    contrast = gray.std()
    
    # Brightness
    brightness = gray.mean()
    
    return {
        'edge_density': edge_density,
        'texture_variance': texture_variance,
        'high_freq_energy': high_freq_energy,
        'contrast': contrast,
        'brightness': brightness
    }


def compute_color_features(frame):
    """Compute color-based features."""
    # Color diversity using histogram entropy
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram for hue channel
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist = hist / hist.sum()  # Normalize
    
    # Entropy (higher = more color diversity)
    hist = hist[hist > 0]  # Remove zeros
    color_entropy = -np.sum(hist * np.log2(hist))
    
    # Saturation statistics
    saturation_mean = hsv[:, :, 1].mean()
    saturation_std = hsv[:, :, 1].std()
    
    return {
        'color_entropy': color_entropy,
        'saturation_mean': saturation_mean,
        'saturation_std': saturation_std
    }


def compute_temporal_features(frame1, frame2):
    """Compute temporal change features."""
    # Frame difference
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    temporal_activity = gray_diff.mean()
    temporal_variance = gray_diff.std()
    
    return {
        'temporal_activity': temporal_activity,
        'temporal_variance': temporal_variance
    }


def extract_video_features(video_path, sample_rate=5):
    """
    Extract comprehensive features from a video.
    
    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame (to speed up computation)
    
    Returns:
        Dictionary of aggregated features
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None
    
    # Storage for frame-level features
    flow_features = []
    spatial_features = []
    color_features = []
    temporal_features = []
    
    prev_frame = None
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample frames
        if frame_count % sample_rate != 0:
            continue
        
        processed_count += 1
        
        # Spatial and color features (every sampled frame)
        spatial_features.append(compute_spatial_features(frame))
        color_features.append(compute_color_features(frame))
        
        # Optical flow and temporal features (requires previous frame)
        if prev_frame is not None:
            flow_features.append(compute_optical_flow_features(prev_frame, frame))
            temporal_features.append(compute_temporal_features(prev_frame, frame))
        
        prev_frame = frame.copy()
    
    cap.release()
    
    if processed_count == 0:
        return None
    
    # Aggregate features across all frames
    aggregated = {
        'total_frames': frame_count,
        'processed_frames': processed_count
    }
    
    # Aggregate flow features
    if flow_features:
        for key in flow_features[0].keys():
            values = [f[key] for f in flow_features]
            aggregated[f'{key}_avg'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_max'] = np.max(values)
    
    # Aggregate spatial features
    for key in spatial_features[0].keys():
        values = [f[key] for f in spatial_features]
        aggregated[f'{key}_avg'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    # Aggregate color features
    for key in color_features[0].keys():
        values = [f[key] for f in color_features]
        aggregated[f'{key}_avg'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    # Aggregate temporal features
    if temporal_features:
        for key in temporal_features[0].keys():
            values = [f[key] for f in temporal_features]
            aggregated[f'{key}_avg'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated


def extract_all_scenes(base_path="outputs", sample_rate=5):
    """Extract features from all 16xSSAA videos."""
    
    all_features = []
    
    print("Extracting features from 16xSSAA videos...")
    print(f"Sample rate: processing every {sample_rate} frames")
    print("="*80)
    
    for scene in tqdm(SCENE_NAMES, desc="Processing scenes"):
        video_path = Path(base_path) / scene / "videos" / "16SSAA.mp4"
        
        if not video_path.exists():
            print(f"\nWarning: {video_path} does not exist, skipping {scene}")
            continue
        
        print(f"\nProcessing {scene}...")
        features = extract_video_features(video_path, sample_rate=sample_rate)
        
        if features:
            features['scene'] = scene
            all_features.append(features)
            print(f"  ✓ Extracted {len(features)} features")
        else:
            print(f"  ✗ Failed to extract features")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns (scene first)
    cols = ['scene'] + [col for col in df.columns if col != 'scene']
    df = df[cols]
    
    return df


if __name__ == "__main__":
    print("="*80)
    print("PHASE 2: VIDEO FEATURE EXTRACTION")
    print("="*80)
    
    # Extract features (sample_rate=5 means every 5th frame for speed)
    # Adjust sample_rate: lower = more accurate but slower, higher = faster but less accurate
    df = extract_all_scenes(sample_rate=5)
    
    # Save to CSV
    output_file = "video_features.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print(f"Saved features to {output_file}")
    print(f"Total scenes processed: {len(df)}")
    print(f"Total features per scene: {len(df.columns) - 1}")
    print("="*80)
    
    # Preview
    print("\nFeature preview (first 3 scenes):")
    print(df.head(3).to_string())
    
    # Feature summary
    print("\n" + "="*80)
    print("Feature Statistics:")
    print("="*80)
    
    # Show key motion features
    motion_cols = [col for col in df.columns if 'flow_mean' in col or 'temporal_activity' in col]
    if motion_cols:
        print("\nMotion Features:")
        print(df[['scene'] + motion_cols].to_string(index=False))
    
    # Show key complexity features
    complexity_cols = [col for col in df.columns if 'edge_density' in col or 'texture_variance' in col]
    if complexity_cols:
        print("\nComplexity Features:")
        print(df[['scene'] + complexity_cols].to_string(index=False))