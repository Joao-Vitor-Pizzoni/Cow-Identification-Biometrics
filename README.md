# Cow-Identification-Biometrics
This project is a Computer Vision and Machine Learning pipeline designed for the individual identification of cattle through muzzle (nose) biometrics.

## Project Overview
Developed as a technical proof of concept for the PECOVISIA project goals, this system automates the process of recognizing animals using deep learning features and classical classifiers.

## Dataset Information

The model was developed using the NewCowMuzzle dataset (version 2025-06-23).

    Source: Hosted on Roboflow Universe, provided by user Fashol.

    Content: Specialized images of cow muzzles annotated for object detection and identification tasks.

    License: Distributed under CC BY 4.0, allowing for academic and commercial use with proper attribution.

    URL: https://universe.roboflow.com/fashol/newcowmuzzle.

## 01_crop_images.py

This script performs the initial data preparation by extracting the Region of Interest (ROI) from the raw dataset.

    Input: Original full-body or head-shot images from the train folder.
    Processing: Using YOLO format labels, the script calculates the precise bounding box coordinates for each animal's muzzle.
    Output: A new, optimized database containing only the cropped muzzle images, effectively removing background noise and focusing on biometric features.
    Processing: Using YOLO format labels, the script calculates the precise bounding box coordinates for each animal's muzzle.

    Output: A new, optimized database containing only the cropped muzzle images, effectively removing background noise and focusing on biometric features.
