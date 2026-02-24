import numpy as np
import OpenEXR
import Imath

class SimpleTAA:
    """
    Simple Temporal Anti-Aliasing - just the basics.
    We'll add features incrementally.
    """
    
    def __init__(self, blend_factor=0.1, disocclusion_threshold=0.1):
        """
        Args:
            blend_factor: Weight of current frame (0-1)
                Lower = more history influence
                Higher = more current frame influence
            disocclusion_threshold: Depth difference threshold for detecting disocclusion
                Smaller = more aggressive disocclusion detection
                Typical values: 0.01 - 0.1
        """
        self.blend_factor = blend_factor
        self.disocclusion_threshold = disocclusion_threshold
        self.history_buffer = None
        self.history_depth = None
        
    def load_exr(self, filepath, channels=['R', 'G', 'B']):
        """Load an EXR file and extract specified channels."""
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_data = []
        
        for channel in channels:
            channel_str = exr_file.channel(channel, pixel_type)
            channel_array = np.frombuffer(channel_str, dtype=np.float32)
            channel_array = channel_array.reshape(height, width)
            channel_data.append(channel_array)
        
        result = np.stack(channel_data, axis=-1)
        return result
    
    def save_exr(self, filepath, image, channels=['R', 'G', 'B']):
        """Save a numpy array as an EXR file."""
        height, width = image.shape[:2]
        
        header = OpenEXR.Header(width, height)
        header['channels'] = {ch: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) 
                              for ch in channels}
        
        out = OpenEXR.OutputFile(filepath, header)
        
        channel_data = {}
        for i, ch in enumerate(channels):
            channel_data[ch] = image[:, :, i].astype(np.float32).tobytes()
        
        out.writePixels(channel_data)
        out.close()
    
    def load_motion_vectors(self, filepath):
        """
        Load motion vectors from EXR file.
        Let's first check what channels are available.
        """
        exr_file = OpenEXR.InputFile(filepath)
        available_channels = list(exr_file.header()['channels'].keys())
        print(f"Available motion vector channels: {available_channels}")
        
        # Try common naming conventions
        possible_names = [
            ['R', 'G'],  # Sometimes saved as R,G channels
            ['X', 'Y'],
            ['motion.x', 'motion.y'],
            ['Velocity.X', 'Velocity.Y'],
            ['U', 'V']
        ]
        
        mv_channels = None
        for names in possible_names:
            if all(ch in available_channels for ch in names):
                mv_channels = names
                print(f"Using motion vector channels: {mv_channels}")
                break
        
        if mv_channels is None:
            raise ValueError(f"Could not find motion vector channels in: {available_channels}")
        
        return self.load_exr(filepath, mv_channels)
    
    def reproject_frame(self, history, motion_vectors):
        """
        Reproject the history buffer using motion vectors.
        
        This is the core of TAA: we use motion vectors to figure out
        where each pixel in the current frame came from in the previous frame.
        """
        height, width = history.shape[:2]
        
        # Create coordinate grid for current frame
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Motion vectors tell us where current pixel came from
        # Subtract to get previous frame coordinates
        prev_x = x_coords - motion_vectors[:, :, 0]
        prev_y = y_coords - motion_vectors[:, :, 1]
        
        # Clamp to valid range (make sure pixel is in image range)
        prev_x = np.clip(prev_x, 0, width - 1)
        prev_y = np.clip(prev_y, 0, height - 1)
        
        # Bilinear interpolation for smooth reprojection
        x0 = np.floor(prev_x).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y0 = np.floor(prev_y).astype(int)
        y1 = np.minimum(y0 + 1, height - 1)
        
        fx = prev_x - x0
        fy = prev_y - y0
        
        # Weights for bilinear interpolation
        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy
        
        # Expand weights for all color channels
        w00 = w00[..., np.newaxis]
        w01 = w01[..., np.newaxis]
        w10 = w10[..., np.newaxis]
        w11 = w11[..., np.newaxis]
        
        # Interpolate
        reprojected = (history[y0, x0] * w00 +
                      history[y1, x0] * w01 +
                      history[y0, x1] * w10 +
                      history[y1, x1] * w11)
        
        return reprojected
    
    def neighborhood_clamp(self, history_color, current_color, kernel_size=3):
        """
        Clamp history color to the min/max range of the current frame's neighborhood.
        
        This is THE key technique for preventing ghosting in TAA.
        
        For each pixel, we:
        1. Look at a small neighborhood (e.g., 3x3) around that pixel in the CURRENT frame
        2. Find the min and max color values in that neighborhood
        3. Clamp the reprojected history sample to lie within [min, max]
        
        This ensures the history can only contribute colors that "make sense" given
        what's currently visible in the scene. If the history is wildly different
        (e.g., from a disoccluded region), it gets clamped to something reasonable.
        
        Args:
            history_color: Reprojected color from previous frame [H, W, 3]
            current_color: Color from current frame [H, W, 3]
            kernel_size: Size of neighborhood (must be odd, typically 3 or 5)
        
        Returns:
            Clamped history color [H, W, 3]
        """
        from scipy.ndimage import minimum_filter, maximum_filter
        
        # Compute min and max in a neighborhood for each channel
        # We need to process each color channel separately
        half_kernel = kernel_size // 2
        
        # Option 1: Use scipy's built-in filters (faster)
        # For each channel, find min/max in the neighborhood
        color_min = np.zeros_like(current_color)
        color_max = np.zeros_like(current_color)
        
        for c in range(current_color.shape[2]):
            color_min[:, :, c] = minimum_filter(current_color[:, :, c], 
                                                 size=kernel_size, 
                                                 mode='nearest')
            color_max[:, :, c] = maximum_filter(current_color[:, :, c], 
                                                 size=kernel_size, 
                                                 mode='nearest')
        
        # Clamp history to [min, max] range
        clamped_history = np.clip(history_color, color_min, color_max)
        
        return clamped_history
    
    def detect_disocclusion(self, current_depth, history_depth, motion_vectors):
        """
        Detect disoccluded pixels where history is invalid.
        
        Disocclusion happens when:
        1. A pixel is newly visible (wasn't in previous frame)
        2. Depth changed significantly (object moved, camera moved)
        3. Motion vector points outside previous frame
        
        We detect this by comparing reprojected history depth with current depth.
        If they differ significantly, the history sample is invalid.
        
        Args:
            current_depth: Depth buffer for current frame [H, W]
            history_depth: Depth buffer from previous frame [H, W]
            motion_vectors: Motion vectors [H, W, 2]
        
        Returns:
            confidence_mask: Float array [H, W] where 1.0 = trust history, 0.0 = reject history
        """
        height, width = current_depth.shape
        
        # Reproject history depth using motion vectors
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        prev_x = x_coords - motion_vectors[:, :, 0]
        prev_y = y_coords - motion_vectors[:, :, 1]
        
        # Check if motion vector points outside frame (disocclusion indicator)
        outside_frame = ((prev_x < 0) | (prev_x >= width) | 
                        (prev_y < 0) | (prev_y >= height))
        
        # Clamp to valid range for sampling
        prev_x = np.clip(prev_x, 0, width - 1)
        prev_y = np.clip(prev_y, 0, height - 1)
        
        # Sample history depth at reprojected locations (using nearest neighbor for depth)
        prev_x_int = prev_x.astype(int)
        prev_y_int = prev_y.astype(int)
        reprojected_depth = history_depth[prev_y_int, prev_x_int]
        
        # Compare depths
        # Unreal's depth is typically in world units or normalized [0,1]
        # We compute relative difference to handle different depth scales
        depth_diff = np.abs(current_depth - reprojected_depth)
        
        # Normalize by current depth to make threshold scale-invariant
        # Add small epsilon to avoid division by zero
        relative_depth_diff = depth_diff / (np.abs(current_depth) + 1e-6)
        
        # Create confidence mask
        # Low confidence (near 0) = disoccluded, should reject history
        # High confidence (near 1) = depth matches, trust history
        confidence = np.ones_like(current_depth)
        
        # Mark pixels with large depth difference as low confidence
        confidence[relative_depth_diff > self.disocclusion_threshold] = 0.0
        
        # Mark pixels that point outside frame as zero confidence
        confidence[outside_frame] = 0.0
        
        return confidence
    
    def process_frame(self, current_color, current_depth, motion_vectors):
        """
        Process a single frame with TAA.
        
        Enhanced algorithm with neighborhood clamping and disocclusion detection:
        1. First frame: initialize history
        2. Other frames: 
           - Reproject history color and depth
           - Detect disocclusion using depth comparison
           - Clamp history to neighborhood bounds (anti-ghosting)
           - Blend with adaptive weight based on disocclusion confidence
        
        Args:
            current_color: Current frame RGB [H, W, 3]
            current_depth: Current frame depth [H, W]
            motion_vectors: Motion vectors [H, W, 2]
        """
        # First frame - initialize history
        if self.history_buffer is None:
            print("First frame - initializing history")
            self.history_buffer = current_color.copy()
            self.history_depth = current_depth.copy()
            return current_color
        
        # Reproject previous frame to current frame's coordinates
        print("Reprojecting history...")
        reprojected_history = self.reproject_frame(self.history_buffer, motion_vectors)
        
        # Detect disocclusion (where history is invalid)
        print("Detecting disocclusion...")
        confidence = self.detect_disocclusion(current_depth, self.history_depth, motion_vectors)
        
        # Apply neighborhood clamping (key anti-ghosting technique)
        print("Applying neighborhood clamping...")
        clamped_history = self.neighborhood_clamp(reprojected_history, current_color)
        
        # Adaptive blending based on confidence
        # Low confidence (disocclusion) -> use more current frame
        # High confidence (valid history) -> use more history
        print(f"Blending with adaptive weights (base factor: {self.blend_factor})")
        
        # Expand confidence to match color channels
        confidence_3d = confidence[..., np.newaxis]
        
        # When confidence is low, increase current frame weight
        # When confidence is high, use original blend factor
        adaptive_current_weight = self.blend_factor + (1 - confidence_3d) * (1 - self.blend_factor)
        
        # Blend
        output = (adaptive_current_weight * current_color + 
                 (1 - adaptive_current_weight) * clamped_history)
        
        # Update history for next frame
        self.history_buffer = output.copy()
        self.history_depth = current_depth.copy()
        
        # Print some stats
        disoccluded_percent = (1 - confidence.mean()) * 100
        print(f"  Disoccluded pixels: {disoccluded_percent:.1f}%")
        
        return output


if __name__ == "__main__":
    import glob
    import os
    import argparse
    import subprocess
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple TAA Implementation')
    parser.add_argument('-alpha', '--blend-factor', type=float, default=0.1,
                        help='Blend factor for current frame (0-1). Lower = more history (stronger AA). Default: 0.1')
    parser.add_argument('-video', '--generate-video', action='store_true',
                        help='Generate video from output frames using ffmpeg')
    parser.add_argument('-fps', '--framerate', type=int, default=30,
                        help='Framerate for output video. Default: 30')
    parser.add_argument('--input', type=str, default="data/fantasticvillage-test/no-taa/",
                        help='Input directory containing EXR frames')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for processed frames (default: auto-generated based on blend factor)')
    
    args = parser.parse_args()
    
    # Your paths
    base_path = args.input
    
    # Auto-generate output directory name based on blend factor if not specified
    if args.output is None:
        output_dir = f"outputs/fantasticvillage-test/taa_alpha{args.blend_factor:.3f}"
    else:
        output_dir = args.output
    
    # Get all color files and sort them
    color_files = sorted(glob.glob(os.path.join(base_path, "FinalImage.*.exr")))
    
    if not color_files:
        print(f"No files found in {base_path}")
        print("Make sure the path is correct!")
    else:
        print(f"Found {len(color_files)} frames")
        print(f"Blend factor (alpha): {args.blend_factor}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize TAA with command line parameters
        # blend_factor: 0.1 = use 90% history (strong AA, more blur)
        #               0.9 = use 90% current (weak AA, sharper)
        # disocclusion_threshold: 0.1 = tolerate 10% relative depth difference
        taa = SimpleTAA(blend_factor=args.blend_factor, disocclusion_threshold=0.1)
        
        # Process each frame
        for i, color_file in enumerate(color_files):
            print(f"\n--- Frame {i+1}/{len(color_files)} ---")
            print(f"Color: {color_file}")
            
            # Extract frame number from color filename
            frame_num = color_file.split('.')[-2]
            
            # Get corresponding files
            motion_file = os.path.join(base_path, f"FinalImagemrq_motionvecs.{frame_num}.exr")
            depth_file = os.path.join(base_path, f"WorldDepth.{frame_num}.exr")
            
            print(f"Motion: {motion_file}")
            print(f"Depth: {depth_file}")
            
            # Check all files exist
            if not os.path.exists(motion_file):
                print(f"WARNING: Motion file not found: {motion_file}")
                continue
            if not os.path.exists(depth_file):
                print(f"WARNING: Depth file not found: {depth_file}")
                continue
            
            # Load data
            current_color = taa.load_exr(color_file, ['R', 'G', 'B'])
            print(f"Color shape: {current_color.shape}")
            
            motion_vectors = taa.load_motion_vectors(motion_file)
            print(f"Motion vectors shape: {motion_vectors.shape}")
            
            # Load depth (usually single channel)
            current_depth = taa.load_exr(depth_file, ['R'])[:, :, 0]  # Take first channel
            print(f"Depth shape: {current_depth.shape}")
            print(f"Depth range: [{current_depth.min():.2f}, {current_depth.max():.2f}]")
            
            # Process
            output = taa.process_frame(current_color, current_depth, motion_vectors)
            
            # Save
            output_file = os.path.join(output_dir, f"taa_frame.{frame_num}.exr")
            taa.save_exr(output_file, output, ['R', 'G', 'B'])
            print(f"Saved: {output_file}")
        
        print("\n=== Done! ===")
        print(f"Output saved to: {output_dir}")
        
        # Generate video if requested
        if args.generate_video:
            print("\n=== Generating Video ===")
            
            # First, convert EXR frames to PNG (ffmpeg handles EXR but PNG is more reliable)
            print("Converting EXR to PNG for video encoding...")
            png_dir = os.path.join(output_dir, "png_frames")
            os.makedirs(png_dir, exist_ok=True)
            
            # Get processed EXR files
            processed_exr_files = sorted(glob.glob(os.path.join(output_dir, "taa_frame.*.exr")))
            
            for i, exr_file in enumerate(processed_exr_files):
                # Load EXR
                img = taa.load_exr(exr_file, ['R', 'G', 'B'])
                
                # Tone map for display (simple gamma correction)
                # You may want to adjust this based on your scene
                img = np.clip(img, 0, 1)  # Clip HDR values
                img = np.power(img, 1/2.2)  # Gamma correction
                
                # Convert to 8-bit
                img_8bit = (img * 255).astype(np.uint8)
                
                # Save as PNG
                from PIL import Image
                png_file = os.path.join(png_dir, f"frame_{i:04d}.png")
                Image.fromarray(img_8bit).save(png_file)
                
                if (i + 1) % 10 == 0:
                    print(f"  Converted {i+1}/{len(processed_exr_files)} frames")
            
            print(f"All frames converted to PNG")
            
            # Generate video using ffmpeg
            video_output = os.path.join(output_dir, f"taa_alpha{args.blend_factor:.3f}_video.mp4")
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', str(args.framerate),
                '-i', os.path.join(png_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',  # High quality (lower = better quality, 18 is visually lossless)
                video_output
            ]
            
            print(f"\nRunning ffmpeg...")
            print(f"Command: {' '.join(ffmpeg_cmd)}")
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                print(f"\n✓ Video generated successfully: {video_output}")
            except subprocess.CalledProcessError as e:
                print(f"\n✗ ffmpeg failed with error:")
                print(e.stderr)
                print("\nMake sure ffmpeg is installed: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)")
            except FileNotFoundError:
                print("\n✗ ffmpeg not found!")
                print("Install ffmpeg: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)")
        
        print("\nTo compare with Unreal's TAA:")
        print("1. Render the same sequence in Unreal with TAA enabled")
        print("2. Compare frame-by-frame visually")
        print("3. Look for: ghosting, edge quality, temporal stability")