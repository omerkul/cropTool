# Quick Start Guide - Screen Detection

## Method 1: Manual Selection (Most Accurate!) ⭐

If you want perfect accuracy, manually mark the screen regions yourself:

```bash
python script.py your_video.mp4 -m
```

**Interactive steps:**
1. A window opens showing a frame from your video
2. **Click and drag** to draw a rectangle around each screen
3. Press **'r'** to reset if you make a mistake
4. Press **'Enter'** when you've marked all screens
5. The script will process using your exact regions

This is the **recommended method** when auto-detection doesn't work perfectly!

## Method 2: Auto-Detection

For videos with non-uniform screen layouts, use auto-detection:

### Step 1: Visualize Screen Detection

First, see what regions the algorithm detects:

```bash
python script.py your_video.mp4 -d -v
```

This will:
- Analyze the video to detect screen boundaries
- Create an image `your_video_regions.jpg` showing the detected screens with colored rectangles
- Print the coordinates and sizes of each detected screen

### Step 2: Process the Video

If the detected regions look correct, process the video:

```bash
python script.py your_video.mp4 -d -o output.mp4
```

If regions aren't correct, use **manual mode** (`-m`) instead!

### Step 3: Analyze Activity (Optional)

To see which screens have the most activity:

```bash
python script.py your_video.mp4 -d -a
```

## How It Works

The auto-detection algorithm uses **3 intelligent strategies**:

### Strategy 1: Black Border Detection
1. **Samples multiple frames** from different parts of the video
2. **Scans for vertical dark lines** (borders/dividers between screens)
3. **Votes across frames** to confirm stable boundaries
4. **Creates regions** based on detected dividers

### Strategy 2: Content Analysis (Smart!)
1. **Analyzes video aspect ratio** to determine possible configurations
2. **Tests each possibility** (2, 3, or 4 screens)
3. **Measures content variance** in each region (different screens = different content)
4. **Checks edge strength** at boundaries (dividers have stronger edges)
5. **Scores each configuration** and picks the best match

### Strategy 3: Edge Detection (Fallback)
1. **Uses Canny edge detection** to find screen boundaries
2. **Finds contours** of rectangular regions
3. **Filters by size** - regions must be 5-95% of total frame
4. **Validates dimensions** - reasonable aspect ratios
5. **Sorts regions** - left to right, top to bottom

The algorithm tries each strategy in order until one succeeds!

## Examples

### Example 1: Manual Selection (Recommended for Accuracy)
```bash
# Interactively mark screen regions yourself
python script.py agent_call.mp4 -m -o clean_output.mp4
```

### Example 2: Auto-Detect with Verification
```bash
# Check what's detected first, then process
python script.py agent_call.mp4 -d -v
# If regions look good:
python script.py agent_call.mp4 -d -o clean_output.mp4
```

### Example 3: Three Different-Sized Screens
```bash
# Manual selection with preview and analysis
python script.py multi_agent.mp4 -m -a -p
```

### Example 4: Equal Split (Simple)
```bash
# For uniformly divided screens
python script.py video.mp4 -n 2 -o output.mp4
```

## Troubleshooting Auto-Detection

### Detection finds wrong regions
- The algorithm looks for rectangular regions with edges
- Make sure screens have visible borders/dividers in the video
- Try adjusting the video to have clearer screen separation

### Detection finds too many/few regions
- The script expects 2-4 screens
- If more regions are detected, the largest/most valid ones are used
- If detection fails completely, it falls back to equal division

### Manual override
If auto-detection doesn't work well, you can always use manual mode:
```bash
python script.py video.mp4 -n 2  # Force 2 equal screens
```

## Tips for Best Results

1. **Always visualize first** (`-v` flag) to verify detection
2. **Use with analysis** (`-a` flag) to see activity distribution
3. **Adjust threshold** if switching is too frequent/infrequent
4. **Increase smoothing** (`-s 10`) for cleaner transitions

## Complete Workflow

```bash
# 1. Visualize detected regions
python script.py video.mp4 -d -v

# 2. Check the generated regions.jpg file

# 3. If regions look good, analyze activity
python script.py video.mp4 -d -a

# 4. Process with appropriate settings
python script.py video.mp4 -d -o final.mp4 -t 1500 -s 8

# 5. Optional: Post-process for better quality
ffmpeg -i final.mp4 -c:v libx264 -crf 18 final_hq.mp4
```

