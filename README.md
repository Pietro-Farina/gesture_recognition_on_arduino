# Gesture Recognition Application Using Feature Extraction on Arduino

## Project Overview

This project implements a real-time gesture recognition system using motion sensor data from an Arduino-compatible IMU (accelerometer and gyroscope).

The system performs:
 - Data collection from IMU sensors
 - Feature extraction (time and frequency domain)
 - Model training using Google Colab
 - Deployment of the trained model on Arduino

The final result is on-device gesture classification in real time.

## Gesture Classes
The system supports multiple gesture classes, such as:
- Rest  
- Shake (left-right)  
- Up-Down  
- Circle
- Punch

The data was collected by sampling 125 values from accelerometer and gyroscope for each axis in a time window of 2 seconds.

## Model Training
The training pipeline includes:
- Data inspection
- Data normalization and feature extraction
- Neural network training  
- Model evaluation

### Feature Extraction
Features are computed using sliding windows over the sensor signal.

We use Time-Domain Features:
- Mean  
- Standard Deviation  
- RMS (Root Mean Square)  
- Minimum  
- Maximum

and Frequency-Domain Features:
- Power Spectral Density (PSD): 3 amplitude and frequencies peaks, 4 spectral bins


## On-Device Inference (Arduino)
The Arduino implementation performs:
- Real-time IMU data acquisition  
- Sliding window buffering  
- Feature computation following the same normalization and extraction pipeline as the one used for training
- Gesture classification using the trained model  
- Output of predicted gesture via Serial Monitor

This enables fully embedded, real-time gesture recognition.

## Considerations
- **Data Quality and Quantity**: The gesture dataset is composed by 20 samples per gesture, collecting more data is the first step to achieve a stronger model. This not only as we would have more data to train the model, but also to better capture the different variation of the gesture (e.g., slow, fast, repeated).
- **Model Performance**: Several networks were tested, and the most common problems were overfitting and failure to generalize one of the possible gesture. Other than having a larger dataset, a possible soultion would be to increase the sampling density within each time window.
