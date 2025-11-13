"""
Active Screen Detector for Multi-Screen Recordings
This script processes videos with 2-4 split screens and creates a single output
showing only the active screen at any given moment.
"""

import os
import platform

# Suppress FFmpeg warnings before importing cv2
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
from collections import deque

# Also set OpenCV log level to silent
cv2.setLogLevel(0)


class ActiveScreenDetector:
    def __init__(self, video_path: str, num_screens: int = 4,
                 activity_threshold: float = 1000, smoothing_window: int = 5,
                 auto_detect: bool = False, manual_select: bool = False,
                 codec: str = 'h264'):
        """
        Initialize the Active Screen Detector

        Args:
            video_path: Path to input video file
            num_screens: Number of screens in the video (2 or 4)
            activity_threshold: Minimum activity level to consider screen active
            smoothing_window: Number of frames to smooth screen selection
            auto_detect: Automatically detect screen regions instead of equal split
            manual_select: Manually select screen regions interactively
            codec: Video codec to use ('h264' or 'mp4v')
        """
        self.video_path = video_path
        self.num_screens = num_screens
        self.activity_threshold = activity_threshold
        self.smoothing_window = smoothing_window
        self.auto_detect = auto_detect
        self.manual_select = manual_select
        self.codec = codec.lower()

        # Open video (FFmpeg warnings are suppressed via environment variable)
        self.cap = cv2.VideoCapture(video_path)

        self.screen_regions = None  # Will hold detected screen boundaries

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps")
        print(f"Total frames: {self.total_frames}")
        print(f"Duration: {self.total_frames / self.fps:.2f} seconds")

        # Manual selection takes priority
        if self.manual_select:
            self.screen_regions = self.manual_region_selection()
            if self.screen_regions:
                self.auto_detect = True  # Use custom regions
                print(f"✅ Manually selected {len(self.screen_regions)} screen regions:")
                for idx, (x, y, w, h) in enumerate(self.screen_regions):
                    print(f"   Screen {idx + 1}: ({x}, {y}) - {w}x{h}")
            else:
                print("⚠️  Manual selection cancelled, falling back to equal split")
                self.manual_select = False
        # Auto-detect screen regions if requested
        elif self.auto_detect:
            print("\n🔍 Auto-detecting screen regions...")
            self.screen_regions = self.detect_screen_regions()
            if self.screen_regions:
                print(f"✅ Detected {len(self.screen_regions)} screen regions:")
                for idx, (x, y, w, h) in enumerate(self.screen_regions):
                    print(f"   Screen {idx + 1}: ({x}, {y}) - {w}x{h}")
            else:
                print("⚠️  Auto-detection failed, falling back to equal split")
                self.auto_detect = False

    def detect_black_borders(self, frame: np.ndarray) -> List[int]:
        """
        Detect vertical black borders/dividers between screens

        Returns:
            List of x-coordinates where vertical dividers are found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Sample vertical line intensities
        vertical_intensities = []
        for x in range(w):
            # Sample middle 80% of height to avoid UI elements at top/bottom
            start_y = int(h * 0.1)
            end_y = int(h * 0.9)
            column_mean = np.mean(gray[start_y:end_y, x])
            vertical_intensities.append(column_mean)

        # Find dark regions (potential borders)
        vertical_intensities = np.array(vertical_intensities)
        threshold = np.percentile(vertical_intensities, 10)  # Bottom 10%

        # Find continuous dark regions
        dark_regions = []
        in_dark = False
        start_x = 0

        for x, intensity in enumerate(vertical_intensities):
            if intensity < threshold:
                if not in_dark:
                    start_x = x
                    in_dark = True
            else:
                if in_dark and (x - start_x) > 5:  # At least 5 pixels wide
                    dark_regions.append((start_x, x))
                in_dark = False

        # Get center of each dark region
        dividers = [int((start + end) / 2) for start, end in dark_regions]

        # Filter dividers that are too close to edges or each other
        min_distance = w * 0.15  # At least 15% from edge or other divider
        filtered_dividers = []

        for div in dividers:
            if div < min_distance or div > w - min_distance:
                continue
            if not filtered_dividers or abs(div - filtered_dividers[-1]) > min_distance:
                filtered_dividers.append(div)

        return filtered_dividers

    def detect_screen_regions(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Automatically detect screen regions using multiple strategies

        Returns:
            List of (x, y, width, height) tuples for each detected screen region
        """
        print("   Strategy 1: Detecting black borders/dividers...")

        # Sample multiple frames to get a stable detection
        sample_frames = []
        frame_indices = [self.total_frames // 4, self.total_frames // 2,
                        3 * self.total_frames // 4]

        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sample_frames.append(frame)

        if not sample_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        # Strategy 1: Detect black borders
        divider_votes = {}
        for frame in sample_frames:
            dividers = self.detect_black_borders(frame)
            for div in dividers:
                # Round to nearest 50 pixels for voting
                div_rounded = (div // 50) * 50
                divider_votes[div_rounded] = divider_votes.get(div_rounded, 0) + 1

        # Get dividers that appear in majority of frames
        confirmed_dividers = [div for div, votes in divider_votes.items()
                            if votes >= len(sample_frames) // 2]
        confirmed_dividers.sort()

        if confirmed_dividers:
            print(f"   Found {len(confirmed_dividers)} vertical dividers")
            # Create regions from dividers
            regions = []
            prev_x = 0

            for div in confirmed_dividers:
                if div - prev_x > 200:  # Minimum screen width
                    regions.append((prev_x, 0, div - prev_x, self.height))
                prev_x = div

            # Add last region
            if self.width - prev_x > 200:
                regions.append((prev_x, 0, self.width - prev_x, self.height))

            if 2 <= len(regions) <= 4:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return regions

        print("   Strategy 2: Analyzing content distribution and activity patterns...")

        # Strategy 2: Intelligent detection based on content analysis
        frame = sample_frames[len(sample_frames) // 2]

        # Test different possible splits and analyze each region
        aspect_ratio = self.width / self.height
        print(f"   Video aspect ratio: {aspect_ratio:.2f}")

        possible_configs = []

        # Determine possible number of screens based on aspect ratio
        # Each screen should have reasonable aspect ratio (0.5 to 2.5)
        if aspect_ratio > 3.0:
            possible_configs.append(4)  # Could be 4 screens
        if aspect_ratio > 2.0:
            possible_configs.append(3)  # Could be 3 screens
        if aspect_ratio > 1.3:
            possible_configs.append(2)  # Could be 2 screens

        if not possible_configs and self.num_screens:
            possible_configs.append(self.num_screens)

        # Analyze each configuration by checking variance in each region
        best_config = None
        best_score = 0

        for num_screens in possible_configs:
            screen_w = self.width // num_screens
            individual_aspect = screen_w / self.height

            # Skip if individual screens would have unreasonable aspect ratio
            if individual_aspect < 0.4 or individual_aspect > 2.8:
                continue

            print(f"   Testing {num_screens}-screen split (each screen: {screen_w}x{self.height}, aspect: {individual_aspect:.2f})...")

            # Calculate variance in each region (more variance = likely separate content)
            total_variance = 0
            for i in range(num_screens):
                x_start = i * screen_w
                x_end = (i + 1) * screen_w
                region = frame[:, x_start:x_end]
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                variance = np.var(gray_region)
                total_variance += variance

            # Average variance per screen
            avg_variance = total_variance / num_screens

            # Check edge strength at boundaries (stronger edges = likely dividers)
            edge_score = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i in range(1, num_screens):
                x = i * screen_w
                # Check +/- 5 pixels around boundary
                if x > 5 and x < self.width - 5:
                    left_mean = np.mean(gray[:, x-5:x])
                    right_mean = np.mean(gray[:, x:x+5])
                    edge_score += abs(left_mean - right_mean)

            # Combined score: content variance + edge strength
            score = avg_variance * 0.7 + edge_score * 0.3

            print(f"      Score: {score:.2f} (variance: {avg_variance:.2f}, edges: {edge_score:.2f})")

            if score > best_score:
                best_score = score
                best_config = num_screens

        if best_config:
            print(f"   ✓ Best match: {best_config} screens (score: {best_score:.2f})")
            screen_w = self.width // best_config
            regions = []
            for i in range(best_config):
                regions.append((i * screen_w, 0, screen_w, self.height))

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return regions

        print("   Strategy 3: Edge detection...")

        # Strategy 3: Original edge detection method
        frame = sample_frames[len(sample_frames) // 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (self.width * self.height) * 0.05
        max_area = (self.width * self.height) * 0.95

        screen_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0 and w > 100 and h > 100:
                    screen_regions.append((x, y, w, h))

        screen_regions.sort(key=lambda r: (r[1] // 100, r[0]))

        if len(screen_regions) >= 2 and len(screen_regions) <= 4:
            print(f"   Edge detection found {len(screen_regions)} regions")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return screen_regions

        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return None

    def split_screens(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Split frame into individual screens based on layout

        Args:
            frame: Input frame to split

        Returns:
            List of screen regions
        """
        # Use auto-detected regions if available
        if self.auto_detect and self.screen_regions:
            screens = []
            for (x, y, w, h) in self.screen_regions:
                # Ensure bounds are within frame
                x = max(0, min(x, frame.shape[1] - 1))
                y = max(0, min(y, frame.shape[0] - 1))
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                # Extract and copy to avoid memory issues
                screen = frame[y:y+h, x:x+w].copy()
                screens.append(screen)

            return screens

        # Otherwise use equal split
        h, w = frame.shape[:2]

        if self.num_screens == 4:
            # 2x2 layout
            screen_h, screen_w = h // 2, w // 2
            screens = [
                frame[0:screen_h, 0:screen_w],              # Top-left
                frame[0:screen_h, screen_w:w],              # Top-right
                frame[screen_h:h, 0:screen_w],              # Bottom-left
                frame[screen_h:h, screen_w:w]               # Bottom-right
            ]
        elif self.num_screens == 2:
            # Side-by-side layout
            screen_w = w // 2
            screens = [
                frame[:, 0:screen_w],                       # Left
                frame[:, screen_w:w]                        # Right
            ]
        elif self.num_screens == 3:
            # Custom 3-screen layout (adjust as needed)
            screen_w = w // 3
            screens = [
                frame[:, 0:screen_w],
                frame[:, screen_w:screen_w*2],
                frame[:, screen_w*2:w]
            ]
        else:
            raise ValueError(f"Unsupported number of screens: {self.num_screens}")

        return screens

    def _adjust_screen_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Adjust manually selected screen regions to fix gaps, overlaps, and snap to edges.
        Limits adjustments to maximum 10 pixels to preserve manual selection accuracy.

        Args:
            regions: List of (x, y, w, h) tuples

        Returns:
            Adjusted list of regions with gaps filled, overlaps fixed, and edges snapped
        """
        if len(regions) == 0:
            return regions

        # Edge snap threshold (pixels) - only snap if very close
        edge_snap_threshold = 10
        # Maximum adjustment allowed
        max_adjustment = 10

        adjusted = []
        sorted_regions = sorted(regions, key=lambda r: r[0])  # Sort by x position

        for i, (x, y, w, h) in enumerate(sorted_regions):
            adj_x, adj_y, adj_w, adj_h = x, y, w, h

            # Snap to left edge if close
            if adj_x <= edge_snap_threshold:
                adj_w = adj_w + adj_x  # Extend width to include the gap
                adj_x = 0
                print(f"      Screen {i+1}: Snapped to left edge (was x={x})")

            # Snap to top edge if close
            if adj_y <= edge_snap_threshold:
                adj_h = adj_h + adj_y  # Extend height to include the gap
                adj_y = 0
                print(f"      Screen {i+1}: Snapped to top edge (was y={y})")

            # Snap to right edge if close
            right_edge = adj_x + adj_w
            if self.width - right_edge <= edge_snap_threshold:
                adj_w = self.width - adj_x
                print(f"      Screen {i+1}: Snapped to right edge (was {right_edge}, now {self.width})")

            # Snap to bottom edge if close
            bottom_edge = adj_y + adj_h
            if self.height - bottom_edge <= edge_snap_threshold:
                adj_h = self.height - adj_y
                print(f"      Screen {i+1}: Snapped to bottom edge (was {bottom_edge}, now {self.height})")

            # Check for gap with previous region (horizontal only for side-by-side screens)
            if i > 0:
                prev_x, prev_y, prev_w, prev_h = adjusted[i-1]
                prev_right = prev_x + prev_w
                gap = adj_x - prev_right

                # Only fix small gaps or overlaps (max 10 pixels)
                if 0 < gap <= max_adjustment:
                    # Small gap - split it between regions
                    split_point = prev_right + gap // 2
                    # Extend previous region to split point
                    adjusted[i-1] = (prev_x, prev_y, split_point - prev_x, prev_h)
                    # Start current region at split point
                    adj_w = adj_w + (adj_x - split_point)
                    adj_x = split_point
                    print(f"      Filled {gap}px gap between Screen {i} and Screen {i+1}")

                elif -max_adjustment <= gap < 0:
                    # Small overlap - fix it
                    overlap = -gap
                    split_point = prev_right + gap // 2
                    # Shrink previous region
                    adjusted[i-1] = (prev_x, prev_y, split_point - prev_x, prev_h)
                    # Start current region at split point
                    adj_w = adj_w + (adj_x - split_point)
                    adj_x = split_point
                    print(f"      Fixed {overlap}px overlap between Screen {i} and Screen {i+1}")

                # If gap/overlap is larger than max_adjustment, don't fix it
                # This preserves the user's manual selection

            # Final boundary check - ensure region is within video bounds
            if adj_x < 0:
                adj_w += adj_x
                adj_x = 0
            if adj_y < 0:
                adj_h += adj_y
                adj_y = 0
            if adj_x + adj_w > self.width:
                adj_w = self.width - adj_x
            if adj_y + adj_h > self.height:
                adj_h = self.height - adj_y

            adjusted.append((adj_x, adj_y, adj_w, adj_h))

        return adjusted

    def _normalize_screen_sizes(self, screens: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize all screens to have the same dimensions by resizing them.
        Uses the largest screen's dimensions or maintains aspect ratios.

        Args:
            screens: List of screen regions with potentially different sizes

        Returns:
            List of screens all resized to the same dimensions
        """
        if not screens:
            return screens

        # Find the maximum width and height across all screens
        max_width = max(screen.shape[1] for screen in screens)
        max_height = max(screen.shape[0] for screen in screens)

        # Alternatively, use the dimensions of the largest screen by area
        # This preserves quality better for the largest screen
        largest_screen = max(screens, key=lambda s: s.shape[0] * s.shape[1])
        target_height, target_width = largest_screen.shape[:2]

        # Resize all screens to match the target dimensions
        normalized_screens = []
        for screen in screens:
            if screen.shape[0] != target_height or screen.shape[1] != target_width:
                # Resize maintaining aspect ratio and padding, or simply resize
                resized = cv2.resize(screen, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                normalized_screens.append(resized)
            else:
                normalized_screens.append(screen)

        return normalized_screens

    def calculate_activity(self, current: np.ndarray, previous: np.ndarray) -> float:
        """
        Calculate activity score between two frames

        Args:
            current: Current frame
            previous: Previous frame

        Returns:
            Activity score (higher = more activity)
        """
        # Convert to grayscale for faster processing
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(curr_gray, prev_gray)

        # Apply threshold to reduce noise
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate activity as sum of differences
        activity = np.sum(thresh)

        return activity

    def detect_active_screen(self, screens: List[np.ndarray],
                            prev_screens: Optional[List[np.ndarray]]) -> int:
        """
        Detect which screen has the most activity

        Args:
            screens: List of current screen regions
            prev_screens: List of previous screen regions

        Returns:
            Index of the most active screen
        """
        if prev_screens is None:
            return 0

        activities = []

        for idx, (curr, prev) in enumerate(zip(screens, prev_screens)):
            activity = self.calculate_activity(curr, prev)
            activities.append(activity)

        # Get screen with maximum activity
        max_idx = int(np.argmax(activities))

        # Check if activity exceeds threshold
        if activities[max_idx] < self.activity_threshold:
            # No significant activity, keep previous screen
            return -1  # Signal to keep previous

        return max_idx

    def process_video(self, output_path: str, show_preview: bool = False):
        """
        Process video and create single-screen output
        Uses ffmpeg for H.264 or OpenCV for mp4v based on codec selection

        Args:
            output_path: Path for output video file
            show_preview: Whether to show live preview (slower)
        """
        import subprocess

        print(f"\nProcessing video...")
        print(f"Output will be saved to: {output_path}")
        print(f"Using codec: {self.codec.upper()}")

        if self.codec == 'h264':
            self._process_video_h264(output_path, show_preview)
        else:  # mp4v
            self._process_video_mp4v(output_path, show_preview)

    def _process_video_h264(self, output_path: str, show_preview: bool = False):
        """
        Process video using H.264 codec via ffmpeg (better compression, slower)
        """
        import subprocess

        # Detect Windows and set appropriate flags
        is_windows = platform.system() == 'Windows'
        creation_flags = 0
        if is_windows:
            # Prevent console window popup on Windows
            creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0x08000000

        # Use ffmpeg to read frames (avoids OpenCV segfaults)
        ffmpeg_read_cmd = [
            'ffmpeg',
            '-i', self.video_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]

        # Start ffmpeg reader with Windows-specific settings
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.DEVNULL,
            'bufsize': 10**8
        }
        if is_windows:
            popen_kwargs['creationflags'] = creation_flags

        ffmpeg_reader = subprocess.Popen(ffmpeg_read_cmd, **popen_kwargs)

        # Read first frame to determine output size
        raw_frame = ffmpeg_reader.stdout.read(self.width * self.height * 3)
        if len(raw_frame) != self.width * self.height * 3:
            ffmpeg_reader.kill()
            raise ValueError("Could not read first frame")

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        screens = self.split_screens(frame)

        # Determine max dimensions across all screens
        max_h = max(screen.shape[0] for screen in screens)
        max_w = max(screen.shape[1] for screen in screens)

        # H.264 requires dimensions divisible by 2
        out_h = max_h if max_h % 2 == 0 else max_h + 1
        out_w = max_w if max_w % 2 == 0 else max_w + 1

        print(f"Output screen size: {out_w}x{out_h}")
        if out_w != max_w or out_h != max_h:
            print(f"   (adjusted from {max_w}x{max_h} to be divisible by 2)")
        print(f"Individual screen sizes:")
        for idx, screen in enumerate(screens):
            print(f"   Screen {idx + 1}: {screen.shape[1]}x{screen.shape[0]}")

        # Create temporary output (no audio)
        temp_output = output_path.replace('.mp4', '_temp.mp4')

        # Start ffmpeg writer for H.264 encoding
        ffmpeg_write_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{out_w}x{out_h}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '28',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            temp_output
        ]

        print("🎬 Starting H.264 encoder...")

        # Start ffmpeg writer with Windows-specific settings
        writer_kwargs = {
            'stdin': subprocess.PIPE,
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.PIPE,
            'bufsize': 10**8 if is_windows else -1  # Large buffer on Windows
        }
        if is_windows:
            writer_kwargs['creationflags'] = creation_flags

        ffmpeg_writer = subprocess.Popen(ffmpeg_write_cmd, **writer_kwargs)

        # Restart reader from beginning
        ffmpeg_reader.kill()
        ffmpeg_reader.wait()
        ffmpeg_reader = subprocess.Popen(ffmpeg_read_cmd, **popen_kwargs)

        # Processing variables
        prev_screens = None
        current_active = 0
        frame_count = 0
        active_buffer = deque(maxlen=self.smoothing_window)

        try:
            while True:
                # Read frame using ffmpeg
                raw_frame = ffmpeg_reader.stdout.read(self.width * self.height * 3)

                if len(raw_frame) != self.width * self.height * 3:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))

                # Split into screens
                screens = self.split_screens(frame)

                # Detect active screen
                active_idx = self.detect_active_screen(screens, prev_screens)

                # Use smoothing buffer
                if active_idx != -1:
                    active_buffer.append(active_idx)

                # Get most common screen in buffer
                if len(active_buffer) > 0:
                    current_active = max(set(active_buffer), key=active_buffer.count)

                # Get active screen
                active_screen = screens[current_active].copy()

                # Resize to match output dimensions if needed
                if active_screen.shape[0] != out_h or active_screen.shape[1] != out_w:
                    active_screen = cv2.resize(active_screen, (out_w, out_h),
                                              interpolation=cv2.INTER_LINEAR)

                # Write to ffmpeg
                try:
                    ffmpeg_writer.stdin.write(active_screen.tobytes())
                except BrokenPipeError:
                    stderr_output = ffmpeg_writer.stderr.read().decode()
                    print(f"\n❌ Encoding process failed: {stderr_output}")
                    break

                # Show preview if requested
                if show_preview:
                    try:
                        cv2.imshow('Active Screen Output', active_screen)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nProcessing cancelled by user")
                            break
                    except:
                        show_preview = False

                # Update progress
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames} frames)", end='\r')

                prev_screens = screens

        finally:
            # Cleanup
            ffmpeg_reader.kill()
            ffmpeg_reader.wait()
            if ffmpeg_writer.stdin:
                ffmpeg_writer.stdin.close()
            ffmpeg_writer.wait()
            if show_preview:
                cv2.destroyAllWindows()

        print(f"\n\n✅ Video processing complete!")
        print(f"Processed {frame_count} frames")

        # Rename temp file to final output
        import shutil
        if os.path.exists(temp_output):
            # Show file size comparison
            input_size = os.path.getsize(self.video_path) / (1024 * 1024)  # MB
            temp_size = os.path.getsize(temp_output) / (1024 * 1024)  # MB
            print(f"\n📦 File size comparison:")
            print(f"   Input:  {input_size:.1f} MB")
            print(f"   Output: {temp_size:.1f} MB")
            print(f"   Ratio:  {temp_size/input_size*100:.1f}% of original")

            shutil.move(temp_output, output_path)
        else:
            print(f"❌ Error: Output file was not created properly")
            return

        # Add audio from original video
        print("\n🎵 Adding audio from original video...")
        final_output = self._add_audio_to_video(output_path)
        if final_output:
            print(f"✅ Final output with audio saved to: {final_output}")
        else:
            print(f"⚠️  Audio merge failed or no audio in source. Video saved to: {output_path}")

    def _process_video_mp4v(self, output_path: str, show_preview: bool = False):
        """
        Process video using mp4v codec via OpenCV (faster, larger files)
        """
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        screens = self.split_screens(frame)

        # Determine max dimensions across all screens
        max_h = max(screen.shape[0] for screen in screens)
        max_w = max(screen.shape[1] for screen in screens)

        print(f"Output screen size: {max_w}x{max_h}")
        print(f"Individual screen sizes:")
        for idx, screen in enumerate(screens):
            print(f"   Screen {idx + 1}: {screen.shape[1]}x{screen.shape[0]}")

        # Create video writer with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (max_w, max_h))

        if not out.isOpened():
            raise ValueError("Could not create video writer")

        print("🎬 Starting mp4v encoder...")

        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Processing variables
        prev_screens = None
        current_active = 0
        frame_count = 0
        active_buffer = deque(maxlen=self.smoothing_window)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Split into screens
                screens = self.split_screens(frame)

                # Detect active screen
                active_idx = self.detect_active_screen(screens, prev_screens)

                # Use smoothing buffer
                if active_idx != -1:
                    active_buffer.append(active_idx)

                # Get most common screen in buffer
                if len(active_buffer) > 0:
                    current_active = max(set(active_buffer), key=active_buffer.count)

                # Get active screen
                active_screen = screens[current_active].copy()

                # Resize to match output dimensions if needed
                if active_screen.shape[0] != max_h or active_screen.shape[1] != max_w:
                    active_screen = cv2.resize(active_screen, (max_w, max_h),
                                              interpolation=cv2.INTER_LINEAR)

                # Write frame
                out.write(active_screen)

                # Show preview if requested
                if show_preview:
                    try:
                        cv2.imshow('Active Screen Output', active_screen)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nProcessing cancelled by user")
                            break
                    except:
                        show_preview = False

                # Update progress
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames} frames)", end='\r')

                prev_screens = screens

        finally:
            out.release()
            if show_preview:
                cv2.destroyAllWindows()

        print(f"\n\n✅ Video processing complete!")
        print(f"Processed {frame_count} frames")

        # Show file size comparison
        input_size = os.path.getsize(self.video_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n📦 File size comparison:")
        print(f"   Input:  {input_size:.1f} MB")
        print(f"   Output: {output_size:.1f} MB")
        print(f"   Ratio:  {output_size/input_size*100:.1f}% of original")

        # Add audio from original video
        print("\n🎵 Adding audio from original video...")
        final_output = self._add_audio_to_video(output_path)
        if final_output:
            print(f"✅ Final output with audio saved to: {final_output}")
        else:
            print(f"⚠️  Audio merge failed or no audio in source. Video saved to: {output_path}")

    def _add_audio_to_video(self, video_path: str) -> Optional[str]:
        """
        Add audio from the original video to the processed video using ffmpeg

        Args:
            video_path: Path to the processed video (without audio)

        Returns:
            Path to the final video with audio, or None if failed
        """
        import subprocess
        import os

        # Create output path for final video with audio
        video_path_obj = Path(video_path)
        final_output = str(video_path_obj.parent / f"{video_path_obj.stem}_with_audio{video_path_obj.suffix}")

        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  ffmpeg not found. Please install ffmpeg to add audio.")
            print("   Install: brew install ffmpeg (macOS) or visit https://ffmpeg.org/")
            return None

        # Use ffmpeg to merge video with audio from original
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,  # Processed video (no audio)
                '-i', self.video_path,  # Original video (with audio)
                '-c:v', 'copy',  # Copy video stream without re-encoding
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0?',  # Use audio from second input (? makes it optional)
                '-shortest',  # Match shortest stream duration
                '-y',  # Overwrite output file
                final_output
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Success! Remove the intermediate video without audio
                os.remove(video_path)
                return final_output
            else:
                print(f"⚠️  ffmpeg error: {result.stderr}")
                return None

        except Exception as e:
            print(f"⚠️  Error adding audio: {str(e)}")
            return None

    def analyze_screens(self, num_samples: int = 100):
        """
        Analyze video to show activity distribution across screens

        Args:
            num_samples: Number of frames to sample for analysis
        """
        print(f"\nAnalyzing video (sampling {num_samples} frames)...")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        screen_activities = [0] * self.num_screens
        sample_interval = max(1, self.total_frames // num_samples)

        prev_screens = None
        samples = 0

        for frame_idx in range(0, self.total_frames, sample_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if not ret:
                break

            screens = self.split_screens(frame)

            if prev_screens is not None:
                for idx, (curr, prev) in enumerate(zip(screens, prev_screens)):
                    activity = self.calculate_activity(curr, prev)
                    screen_activities[idx] += activity

            prev_screens = screens
            samples += 1

        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Print analysis results
        print("\n📊 Activity Analysis:")
        print("-" * 50)
        total_activity = sum(screen_activities)

        for idx, activity in enumerate(screen_activities):
            percentage = (activity / total_activity * 100) if total_activity > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"Screen {idx + 1}: {bar} {percentage:.1f}%")

        print("-" * 50)

    def manual_region_selection(self) -> List[Tuple[int, int, int, int]]:
        """
        Interactive manual selection of screen regions

        Returns:
            List of (x, y, width, height) tuples for each manually selected region
        """
        print("\n📍 Manual Region Selection Mode")
        print("=" * 60)
        print("Instructions:")
        print("  1. Click and drag to draw a rectangle around each screen")
        print("  2. Press 'r' to reset if you make a mistake")
        print("  3. Press 'Enter' when done selecting all screens")
        print("  4. Press 'q' to cancel")
        print("=" * 60)

        # Get a frame from the middle of the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames // 2)
        ret, frame = self.cap.read()

        if not ret:
            print("❌ Could not read frame")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        # Resize frame if too large for screen
        display_frame = frame.copy()
        scale_factor = 1.0
        max_display_width = 3840
        max_display_height = 2160

        if frame.shape[1] > max_display_width or frame.shape[0] > max_display_height:
            scale_w = max_display_width / frame.shape[1]
            scale_h = max_display_height / frame.shape[0]
            scale_factor = min(scale_w, scale_h)
            new_w = int(frame.shape[1] * scale_factor)
            new_h = int(frame.shape[0] * scale_factor)
            display_frame = cv2.resize(frame, (new_w, new_h))
            print(f"   Frame scaled to {new_w}x{new_h} for display (scale: {scale_factor:.2f})")

        regions = []
        current_rect = None
        drawing = False
        start_point = None

        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, current_rect, regions

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                current_rect = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    current_rect = (start_point[0], start_point[1],
                                   x - start_point[0], y - start_point[1])

            elif event == cv2.EVENT_LBUTTONUP:
                if drawing and start_point:
                    drawing = False
                    x1, y1 = start_point
                    w, h = x - x1, y - y1

                    # Normalize rectangle (handle dragging in any direction)
                    if w < 0:
                        x1 = x1 + w
                        w = -w
                    if h < 0:
                        y1 = y1 + h
                        h = -h

                    # Only add if region is large enough
                    if w > 50 and h > 50:
                        # Scale back to original coordinates
                        orig_x = int(x1 / scale_factor)
                        orig_y = int(y1 / scale_factor)
                        orig_w = int(w / scale_factor)
                        orig_h = int(h / scale_factor)

                        regions.append((orig_x, orig_y, orig_w, orig_h))
                        print(f"   Screen {len(regions)} added: ({orig_x}, {orig_y}) - {orig_w}x{orig_h}")
                        current_rect = None

        cv2.namedWindow('Select Screen Regions')
        cv2.setMouseCallback('Select Screen Regions', mouse_callback)

        while True:
            # Draw the frame with regions
            temp_frame = display_frame.copy()

            # Draw completed regions
            for idx, (rx, ry, rw, rh) in enumerate(regions):
                # Scale coordinates for display
                disp_x = int(rx * scale_factor)
                disp_y = int(ry * scale_factor)
                disp_w = int(rw * scale_factor)
                disp_h = int(rh * scale_factor)

                color = colors[idx % len(colors)]
                cv2.rectangle(temp_frame, (disp_x, disp_y),
                            (disp_x + disp_w, disp_y + disp_h), color, 2)

                # Add label
                label = f"Screen {idx + 1}"
                cv2.putText(temp_frame, label, (disp_x + 5, disp_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw current rectangle being drawn
            if current_rect:
                x, y, w, h = current_rect
                if w < 0:
                    x = x + w
                    w = -w
                if h < 0:
                    y = y + h
                    h = -h
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Add instructions overlay
            instruction_text = f"Screens: {len(regions)} | Press 'r' to reset, 'Enter' to finish, 'q' to cancel"
            cv2.putText(temp_frame, instruction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(temp_frame, instruction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            cv2.imshow('Select Screen Regions', temp_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("   Selection cancelled")
                cv2.destroyAllWindows()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None

            elif key == ord('r'):
                print("   Resetting selections")
                regions = []

            elif key == 13 or key == 10:  # Enter key
                if len(regions) >= 2:
                    print(f"   ✅ {len(regions)} regions selected")
                    break
                else:
                    print("   ⚠️  Please select at least 2 screen regions")

        cv2.destroyAllWindows()

        # Sort regions left to right, top to bottom
        regions.sort(key=lambda r: (r[1] // 100, r[0]))

        # Adjust regions to fix gaps and overlaps
        print("   🔧 Auto-adjusting regions to fix gaps and overlaps...")
        original_regions = regions.copy()
        regions = self._adjust_screen_regions(regions)

        # Show adjustments made
        for i, ((ox, oy, ow, oh), (nx, ny, nw, nh)) in enumerate(zip(original_regions, regions)):
            if (ox, oy, ow, oh) != (nx, ny, nw, nh):
                print(f"      Screen {i+1}: ({ox}, {oy}, {ow}x{oh}) → ({nx}, {ny}, {nw}x{nh})")

        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return regions

    def visualize_regions(self, output_path: str = "screen_regions.jpg"):
        """
        Create a visualization of detected screen regions

        Args:
            output_path: Path to save the visualization image
        """
        if not self.screen_regions:
            print("⚠️  No screen regions to visualize")
            return

        # Get a frame from the middle of the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames // 2)
        ret, frame = self.cap.read()

        if not ret:
            print("❌ Could not read frame for visualization")
            return

        # Draw rectangles around detected regions
        vis_frame = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

        for idx, (x, y, w, h) in enumerate(self.screen_regions):
            color = colors[idx % len(colors)]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 3)

            # Add label
            label = f"Screen {idx + 1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(vis_frame, (x, y - label_size[1] - 10),
                         (x + label_size[0] + 10, y), color, -1)
            cv2.putText(vis_frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save visualization
        cv2.imwrite(output_path, vis_frame)
        print(f"✅ Screen regions visualization saved to: {output_path}")

        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def main():
    parser = argparse.ArgumentParser(
        description='Process multi-screen recordings to show only the active screen'
    )
    parser.add_argument('input', type=str, help='Input video file path')
    parser.add_argument('-o', '--output', type=str, help='Output video file path')
    parser.add_argument('-n', '--num-screens', type=int, default=4,
                       choices=[2, 3, 4], help='Number of screens in video (default: 4)')
    parser.add_argument('-t', '--threshold', type=float, default=1000,
                       help='Activity threshold (default: 1000)')
    parser.add_argument('-s', '--smoothing', type=int, default=5,
                       help='Smoothing window size (default: 5)')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Show live preview during processing')
    parser.add_argument('-a', '--analyze', action='store_true',
                       help='Analyze video before processing')
    parser.add_argument('-d', '--auto-detect', action='store_true',
                       help='Automatically detect screen regions (for non-uniform layouts)')
    parser.add_argument('-m', '--manual', action='store_true',
                       help='Manually select screen regions interactively')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Visualize detected screen regions and save as image')
    parser.add_argument('-c', '--codec', type=str, default='h264',
                       choices=['h264', 'mp4v'],
                       help='Video codec to use: h264 (better compression, slower) or mp4v (faster, larger files). Default: h264')

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return

    # Generate output filename if not provided
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.parent / f"{input_path.stem}_cropped.mp4")

    try:
        # Create detector
        detector = ActiveScreenDetector(
            video_path=str(input_path),
            num_screens=args.num_screens,
            activity_threshold=args.threshold,
            smoothing_window=args.smoothing,
            auto_detect=args.auto_detect,
            manual_select=args.manual,
            codec=args.codec
        )

        # Optional visualization
        if args.visualize:
            vis_path = str(input_path.parent / f"{input_path.stem}_regions.jpg")
            detector.visualize_regions(vis_path)
            print("\n")

        # Optional analysis
        if args.analyze:
            detector.analyze_screens()
            print("\n")

        # Process video
        detector.process_video(output_path, show_preview=args.preview)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # If no arguments provided, show help
    import sys
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Active Screen Detector - Multi-Screen Video Processor")
        print("=" * 60)
        print("\nUsage Examples:")
        print("  python script.py input_video.mp4")
        print("  python script.py input_video.mp4 -m  # Manual selection")
        print("  python script.py input_video.mp4 -d -v  # Auto-detect & visualize")
        print("  python script.py input_video.mp4 -n 2")
        print("  python script.py input_video.mp4 -o output.mp4 -n 4 -a")
        print("  python script.py input_video.mp4 -d -p -t 2000")
        print("\nOptions:")
        print("  -n, --num-screens    Number of screens (2, 3, or 4)")
        print("  -o, --output         Output file path")
        print("  -t, --threshold      Activity detection threshold")
        print("  -s, --smoothing      Smoothing window size")
        print("  -p, --preview        Show live preview")
        print("  -a, --analyze        Analyze video first")
        print("  -d, --auto-detect    Auto-detect screen regions (non-uniform layouts)")
        print("  -m, --manual         Manually select screen regions (most accurate!)")
        print("  -v, --visualize      Show detected screen regions")
        print("\nFor full help: python script.py --help")
        print("=" * 60)
    else:
        main()
