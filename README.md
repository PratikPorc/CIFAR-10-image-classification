# CIFAR-10 Image Classification with CNN

This project is a deep learning-based image classifier built using the **CIFAR-10 dataset**, a collection of 60,000 32x32 color images in 10 different classes. The model is implemented using **TensorFlow** and **Keras**, and is trained to classify images into categories like airplanes, cars, birds, cats, and more.

## 🔍 Overview

The CIFAR-10 dataset includes 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

This project builds and trains a Convolutional Neural Network (CNN) that can classify unseen images into one of these 10 categories.

## 🛠️ Tech Stack

- Python
- TensorFlow & Keras
- NumPy
- Matplotlib

## 🧠 Model Architecture

- **Input Layer**: 32x32x3 RGB image
- **Convolutional Layers** with ReLU activation
- **MaxPooling Layers** to downsample
- **Dropout** for regularization
- **Dense Layers** leading to a softmax output layer (10 classes)

## 📦 Dataset

- Dataset: CIFAR-10
- Size: 60,000 images (50,000 train, 10,000 test)
- Shape: (32, 32, 3)

## 📊 Training Details

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metrics: Accuracy
- Epochs: (customizable based on your configuration)
- Visualization: Training vs Validation Accuracy and Loss

## 📈 Results

- Achieved good accuracy on the test set
- Visualized sample predictions
- Plotted training history using Matplotlib

