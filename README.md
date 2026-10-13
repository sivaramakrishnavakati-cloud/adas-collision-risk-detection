# ADAS Computer Vision and Collision Risk Indicator

## Overview

This project explores computer vision techniques used in Advanced Driver Assistance Systems (ADAS).

It uses YOLO object detection and multi-object tracking to identify vehicles in driving footage, follow their positions across video frames, and calculate an experimental approach indicator based on bounding-box area growth.

The project also records observations in CSV format and generates evaluation reports and visualizations.

## Features

- Vehicle detection using YOLO.
- Multi-object tracking using ByteTrack.
- Tracking IDs for detected vehicles.
- Bounding-box area analysis over time.
- Experimental approach indicator.
- CSV logging of vehicle observations.
- Evaluation reports using pandas.
- Charts showing indicator observations by vehicle class and status.

## Technologies

- Python
- Ultralytics YOLO
- OpenCV
- ByteTrack
- pandas
- Matplotlib

## Project Structure

```text
adas-collision-risk-detection/
├── collision_risk.py
├── track_vehicles.py
├── detect_objects.py
├── evaluate_results.py
├── visualize_results.py
├── requirements.txt
├── .gitignore
└── outputs/
    ├── approach_indicator_log.csv
    ├── evaluation_report.txt
    ├── indicator_status_chart.png
    └── vehicle_class_chart.png