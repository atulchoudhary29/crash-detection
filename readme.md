# Crash Detection

This repository provides a crash detection solution using the YOLOv4 object detection model. The program detects vehicles in real-time using a webcam or a pre-recorded video and determines if a crash has occurred based on their distance and velocity.

## Table of Contents

1. [Installation](#install)
2. [Usage](#usage)
3. [Examples](#examples)
  
 <a name="install"></a> 

  ## 1. Installation
1. Clone the repository:

	```
	git clone https://github.com/your_username/crash-detection.git
	```
    
2. Install the required packages:

	```
	pip install -r requirements.txt
	```

3. Download the YOLOv4 Tiny weights and cfg files:
  
	 -   [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
	-   [yolov4-tiny.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)
	
	Save these files in the project directory.
  
<a name="usage"></a>  
## 2. Usage
  
The program can be run using either a webcam or a pre-recorded video file as input. Use the following command-line arguments to specify the desired mode:

```
usage: main.py [-h] [--webcam WEBCAM] [--video_path VIDEO_PATH] [--verbose VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  --webcam WEBCAM       True/False
  --video_path VIDEO_PATH   Path of video file
  --verbose VERBOSE     To print statements
```

-   `--webcam`: Set to True to use a webcam as input (default: False).
-   `--video_path`: Path to the video file to be processed (default: "video.mp4").

### Webcam Mode

To run the crash detection program using your webcam, use the following command:
```
python main.py --webcam True
```

### Video Mode

To run the crash detection program using a video file, use the following command:

```
python main.py --video_path <path_to_video_file>
```

Replace `<path_to_video_file>` with the path to your video file.

<a name="examples"></a> 
## 3. Examples

Detect crashes in a video file named "traffic.mp4":

```
python crash_detection.py --video_path traffic.mp4
```