# Active Screen Detector

A Python tool for processing multi-screen recording videos into single-screen outputs by automatically detecting and showing only the active screen at any given moment.

## 🎯 Features

- **Automatic Active Screen Detection**: Uses computer vision to detect which screen has the most activity
- **Auto-Detect Screen Regions**: Automatically detect non-uniform screen layouts (different sizes/positions)
- **Multiple Screen Layouts**: Supports 2, 3, or 4 screen configurations
- **Temporal Smoothing**: Prevents rapid screen switching with configurable smoothing window
- **Activity Analysis**: Analyze videos to see activity distribution across screens
- **Visual Region Detection**: Visualize detected screen boundaries before processing
- **Live Preview**: Optional real-time preview during processing
- **Customizable Parameters**: Adjust detection sensitivity and smoothing

## 📋 Requirements

- Python 3.7+
- OpenCV
- NumPy

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

Process a 4-screen video (default):
```bash
python script.py input_video.mp4
```

### Auto-Detect Non-Uniform Layouts

**For videos with screens of different sizes or positions:**
```bash
python script.py input_video.mp4 -d
```

Visualize detected regions first:
```bash
python script.py input_video.mp4 -d -v
```

This creates an image showing the detected screen boundaries before processing.

### Manual Region Selection (Most Accurate!)

**For precise control, manually select screen regions interactively:**
```bash
python script.py input_video.mp4 -m
```

This opens an interactive window where you can:
1. Click and drag to draw rectangles around each screen
2. Press 'r' to reset if you make a mistake
3. Press 'Enter' when done selecting all screens
4. Press 'q' to cancel

Manual selection is recommended when auto-detection doesn't work well or when you need exact boundaries.

### Specify Number of Screens

For 2-screen layout:
```bash
python script.py input_video.mp4 -n 2
```

For 3-screen layout:
```bash
python script.py input_video.mp4 -n 3
```

### Custom Output Path

```bash
python script.py input_video.mp4 -o output_video.mp4
```

### Analyze Video First

See which screens have the most activity before processing:
```bash
python script.py input_video.mp4 -a
```

### Live Preview

Watch the processing in real-time (press 'q' to cancel):
```bash
python script.py input_video.mp4 -p
```

### Advanced Options

Adjust detection sensitivity and smoothing:
```bash
python script.py input_video.mp4 -t 2000 -s 10
```

- `-t, --threshold`: Activity threshold (higher = less sensitive, default: 1000)
- `-s, --smoothing`: Smoothing window frames (higher = smoother transitions, default: 5)

### Complete Example

```bash
python script.py agent_recording.mp4 -n 4 -o processed_output.mp4 -a -t 1500 -s 8
```

## 🎬 How It Works

1. **Screen Detection** (optional): Uses edge detection and contour analysis to find screen boundaries
2. **Frame Splitting**: Each frame is divided into individual screen regions based on the layout
3. **Activity Detection**: Compares consecutive frames to detect pixel changes in each screen
4. **Screen Selection**: Selects the screen with the highest activity level
5. **Temporal Smoothing**: Uses a sliding window to prevent rapid switching between screens
6. **Video Output**: Creates a single-screen video showing only the active screen

## 🔧 Algorithm Details

The detector uses:
- **Auto-Detection** (when enabled):
  - Edge detection (Canny algorithm) to find screen boundaries
  - Contour analysis to identify rectangular regions
  - Filters by area, aspect ratio, and size to find valid screens
  - Samples multiple frames for stable detection
- **Grayscale conversion** for faster processing
- **Frame differencing** to detect motion and changes
- **Threshold filtering** to reduce noise
- **Activity scoring** based on pixel change sum
- **Mode-based smoothing** to ensure stable screen selection

## 📊 Screen Layouts

### 4-Screen Layout (2x2)
```
┌─────────┬─────────┐
│ Screen 1│ Screen 2│
│ (Top-L) │ (Top-R) │
├─────────┼─────────┤
│ Screen 3│ Screen 4│
│ (Bot-L) │ (Bot-R) │
└─────────┴─────────┘
```

### 2-Screen Layout (Side-by-Side)
```
┌─────────┬─────────┐
│         │         │
│ Screen 1│ Screen 2│
│  (Left) │ (Right) │
│         │         │
└─────────┴─────────┘
```

### 3-Screen Layout (Horizontal)
```
┌────────┬────────┬────────┐
│        │        │        │
│Screen 1│Screen 2│Screen 3│
│        │        │        │
└────────┴────────┴────────┘
```

## 🎛️ Parameters Tuning

### Activity Threshold (`-t`)
- **Lower values (500-1000)**: More sensitive, switches screens more frequently
- **Default (1000)**: Balanced for typical recordings
- **Higher values (2000-5000)**: Less sensitive, only switches on significant activity

### Smoothing Window (`-s`)
- **Lower values (2-3)**: Quick response, may switch frequently
- **Default (5)**: Good balance between responsiveness and stability
- **Higher values (10-15)**: Very smooth, slower to switch screens

## 📈 Example Workflow

1. **First, analyze your video:**
   ```bash
   python script.py agent_video.mp4 -a
   ```
   
2. **Review the activity distribution to understand your video**

3. **Process with appropriate settings:**
   ```bash
   python script.py agent_video.mp4 -n 4 -o clean_output.mp4
   ```

4. **If switching is too frequent, increase smoothing:**
   ```bash
   python script.py agent_video.mp4 -s 10 -o clean_output.mp4
   ```

5. **If it's not detecting activity well, adjust threshold:**
   ```bash
   python script.py agent_video.mp4 -t 500 -o clean_output.mp4
   ```

## 🐛 Troubleshooting

### Video not loading
- Ensure the video file path is correct
- Try using absolute paths
- Check that the video codec is supported by OpenCV

### No screen switching detected
- Lower the threshold: `-t 500`
- Check if your video actually has activity on different screens
- Use `-a` to analyze activity distribution

### Too much screen switching
- Increase smoothing: `-s 10` or higher
- Increase threshold: `-t 2000` or higher

### Output video quality issues
- The script uses 'mp4v' codec; for better quality, consider post-processing with ffmpeg:
  ```bash
  ffmpeg -i output.mp4 -c:v libx264 -crf 18 -preset slow final_output.mp4
  ```

## 🎯 Use Cases

- **Call Center Recordings**: Process multi-monitor agent recordings
- **Technical Support**: Capture only active screens from tech support sessions
- **Training Videos**: Create cleaner training materials from multi-screen captures
- **Screen Sharing**: Clean up multi-monitor screen shares for presentations

## 📝 Notes

- The output video will have "Screen X" text overlay showing which screen is active
- Processing time depends on video length and resolution
- Higher resolution videos take longer to process
- Preview mode (`-p`) slows down processing but lets you monitor progress

## 🔮 Future Enhancements

Potential improvements:
- Mouse cursor detection for smarter screen selection
- Web UI for easier usage
- Batch processing multiple videos
- Audio processing to sync with visual activity
- Custom screen region definitions
- GPU acceleration for faster processing

## 📄 License

Free to use and modify for your needs.

