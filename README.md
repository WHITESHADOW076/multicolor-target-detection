# Target Detection using Python and OpenCV

A simple **color-based target detection** system built with **Python** and **OpenCV**.  
Detects multiple targets (Red, Green, Blue) in real-time using a webcam or video input.



## Features

- Detects **Red**, **Green**, and **Blue** targets simultaneously  
- Draws bounding boxes and centered markers  
- Displays live video feed with color labels  
- Works with **webcam** or **video files**  
- Easy to extend for more colors or shapes  



## Installation

Make sure you have **Python 3** installed, then run:

```bash
pip install opencv-python numpy
```



## Usage
- Run the script:

```bash
python multi_color_target_detection.py
```
- A webcam window will open showing detected colored targets

- Press ESC to exit



## How It Works
1. Converts frames to HSV color space for better segmentation

2. Applies color masks for Red, Green, and Blue

3. Removes noise using morphological operations

4. Finds contours and draws bounding boxes

5. Displays the annotated video in real-time
