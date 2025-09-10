# ORB-AR-video-mapper
Augmented Reality project using feature detection and video overlay.
Marker-Based Augmented Reality using OpenCV

Project Overview

This project demonstrates a marker-based Augmented Reality (AR) system built using Python and OpenCV.
It detects a target image in the live webcam feed and overlays a video on top of it in real time.

The system uses:

ORB (Oriented FAST and Rotated BRIEF) for feature detection and description.

BFMatcher for matching keypoints between the target image and webcam feed.

Homography transformation to warp the video onto the detected region.

Smoothing to reduce jitter for a stable AR experience.



Features

Detects a predefined target image in real time.

Overlays a video file (video.mp4) onto the detected target area.

Smooths the transformation matrix to avoid jitter.

Displays FPS (frames per second) for performance monitoring.

Shows status messages (“Target Found” / “Target Not Found”).

Automatically resets video playback when the target is lost.
