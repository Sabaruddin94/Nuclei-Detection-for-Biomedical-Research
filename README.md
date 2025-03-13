Nuclei Detection for Biomedical Research

This project involves the development of a semantic segmentation model for detecting cell nuclei in biomedical images. 
The tool aims to expedite research in understanding various diseases such as cancer, Alzheimer's, and heart disease by automating the process of identifying cell nuclei in microscopic images. 
The model uses a U-Net architecture with transfer learning to segment images and identify the regions of interest (nuclei), which are essential for further genetic analysis and drug testing.

Project Overview
The goal of this project is to build an efficient and accurate model for nuclei segmentation using a deep learning approach. 
This project is part of a larger effort to improve the speed and accuracy of disease research by automating the detection of key biological features in medical images.

Key Steps
1. Data Preparation: Load, preprocess, and augment image and mask data from the Data Science Bowl 2018 dataset.
2. Model Development: Use U-Net architecture with a MobileNetV2 backbone for feature extraction.
3. Training: Train the model with proper validation to avoid overfitting, utilizing techniques such as early stopping and model checkpointing.
4. Evaluation: Evaluate the model on the test set, achieving more than 80% accuracy.

Requirements
Python 3.x
TensorFlow 2.x
Keras
OpenCV
Matplotlib
Scikit-learn
TensorFlow Datasets

Dataset
The dataset used for training and evaluation is the Data Science Bowl 2018 dataset, which contains images of cell nuclei and corresponding segmentation masks. 
The dataset can be downloaded from Kaggle:Data Science Bowl 2018 Dataset.
Link to the source of data: https://www.kaggle.com/competitions/data-science-bowl-2018/overview

Model Architecture
The model is built using the U-Net architecture, which is highly effective for image segmentation tasks. The model consists of:

1. Downsampling Path: Extracts relevant features using a pretrained MobileNetV2 model.
2. Upsampling Path: Restores the spatial dimensions and refines the segmentation.
3. Output Layer: A single convolutional layer with a sigmoid activation for binary segmentation (nucleus vs. background).
   
Code Structure
1. Data Loading & Preprocessing: Scripts to load images and masks, resize, normalize, and augment the data.
2. Model Definition: The U-Net model, using MobileNetV2 for feature extraction.
3. Training: Training the model, monitoring for overfitting using TensorBoard and EarlyStopping.
4. Evaluation: Evaluating the model on test data.
   
Training & TensorBoard Integration
During training, TensorBoard is used to monitor model performance in real-time. This provides useful visualizations such as loss curves, accuracy, and other metrics.

Here’s a sample screenshot from TensorBoard showing the training and validation loss and accuracy.
![alt text](<Image/epoch accuracy.png>)

![alt text](<Image/epocj loss.png>)

Usage
1. Data Preprocessing: Follow the data_preprocessing.py script to prepare the data for training.
2. Model Training: Train the model using the train.ipynb script.
3. Model Evaluation: After training, evaluate the model with evaluate_model.ipynb.
4. Prediction: Use the trained model to predict the segmentation mask on new images.

Results
The model achieves a validation accuracy of >80%, demonstrating its potential to be used in biomedical research applications for identifying cell nuclei in medical images.

Future Work
1. Hyperparameter Tuning: Further optimization of the model hyperparameters to improve accuracy.
2. Model Improvements: Incorporating additional advanced techniques like attention mechanisms or more sophisticated augmentations.
3. Deployment: Deploying the model into a web-based application for easy access and use by researchers.
