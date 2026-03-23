# Cow-Identification-Biometrics
This project is a Computer Vision and Machine Learning pipeline designed for the individual identification of cattle through muzzle (nose) biometrics.

## Project Overview
Developed as a technical proof of concept for the PECOVISIA project goals, this system automates the process of recognizing animals using deep learning features and classical classifiers.

## Dataset Information
The model was developed using the NewCowMuzzle dataset (version 2025-06-23).

- Source: Hosted on Roboflow Universe, provided by user Fashol.
- Content: Specialized images of cow muzzles annotated for object detection and identification tasks.
- License: Distributed under CC BY 4.0, allowing for academic and commercial use with proper attribution.
- URL: https://universe.roboflow.com/fashol/newcowmuzzle.

## 01_crop_images.py
This script performs the initial data preparation by extracting the Region of Interest (ROI) from the raw dataset.

-Input: Original full-body or head-shot images from the train folder.
-Processing: Using YOLO format labels, the script calculates the precise bounding box coordinates for each animal's muzzle.
-Output: A new, optimized database containing only the cropped muzzle images, effectively removing background noise and focusing on biometric features.
-Processing: Using YOLO format labels, the script calculates the precise bounding box coordinates for each animal's muzzle.

## Original Image (Full View)	Processed Image (Muzzle Crop)
<img width="487" height="648" alt="image" src="https://github.com/user-attachments/assets/ff95cf15-aed0-4d8b-b45e-67c04b18626c" />

## Original image with YOLO label	Cropped ROI ready for CNN
<img width="238" height="283" alt="image" src="https://github.com/user-attachments/assets/611bd069-40aa-47dd-b1ef-dafac40da31a" />

## 02_extract_embeddings.py
This script transforms the cropped muzzle images into a high-dimensional digital signature (embedding).

- Architecture: Uses a pre-trained ResNet18 (CNN) as the backbone feature extractor.
- Method: The final fully connected layer of the model is replaced with an Identity layer to output raw features instead of class labels.
- Result: Each muzzle image is converted into a 512-dimensional vector that represents its unique biometric pattern.

## 03_inference_matcher.py
This script represents the core identification engine of the project, designed to recognize an individual animal from a single "unseen" image.

- Goal: To perform real-time biometric matching by comparing a new muzzle photo against the registered database.
- Feature Extraction: It processes the input image through the ResNet18 CNN to generate its unique 512-dimensional embedding.
- Similarity Analysis: The script calculates the Cosine Similarity between the new embedding and every entry in the embeddings_vacas.npy database.
- Identification Output: It identifies the most likely candidate by finding the highest similarity score, returning both the Cow ID (filename) and the Confidence Level.
- Validation Tool: This serves as the primary tool for validating the identification solutions proposed in the PECOVISIA project framework.

## Data Insight & Organization Strategy
Upon analyzing the raw dataset, I identified that many images belonged to the same individuals, estimated at approximately 200 distinct cows within the training set. To optimize the identification process and ensure a structured database for supervised learning, I implemented an automated organization strategy.

## 04_clustering_identification.py
- Context: Instead of manually labeling hundreds of images, I developed this script to deduce identities based on visual similarity.
- Method: Using K-Means Clustering on the extracted 512-dimensional embeddings, the system groups images that share the same biometric patterns.
- Outcome: The script automatically re-organizes and renames the files (e.g., 1_01.jpg, 1_02.jpg), creating a clean, labeled dataset from previously unorganized data.
- Impact: This step ensures that the subsequent SVM and Random Forest models are trained on accurate, high-quality clusters of individual animals.

## 05_prepare_training_data.py 
After organizing the images into clusters, this script structures the final dataset for the supervised learning phase.

- Goal: To create a consolidated training file containing all biometric features (X) and their respective cow identities (y).
- Data Extraction: The script iterates through the newly organized folders (e.g., Cow_1, Cow_2) and extracts the ID from the filename.
- Feature Mapping: It uses the ResNet18 CNN to generate the final embeddings for every image in the structured dataset.
- Output: Saves the feature matrix and labels into a single compressed NumPy file (dados_treinamento_vaca.npz).

## 06_train_classifiers.py
This final stage of the pipeline focuses on benchmarking different supervised learning algorithms to identify the most accurate model for bovine recognition.
   
- Goal: To train and compare multiple classification models using the structured biometric embeddings.
- Algorithms: Implements and evaluates Random Forest (100 estimators) and Support Vector Machine (SVM) with a linear kernel.
- Validation Strategy: Uses an 80/20 Train-Test split to ensure the models can accurately identify "new" images of cows they haven't seen during training.
- Performance Visualization: Automatically generates a comparison bar chart (resultado_acuracia.png) showing the accuracy percentage for each model.

<img width="803" height="503" alt="image" src="https://github.com/user-attachments/assets/aea8997e-4d63-4991-8231-11dac1aedf76" />
