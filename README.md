# Bird Audio Classification Project

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Audio Processing](#audio-processing)
4. [Feature Extraction](#feature-extraction)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Geospatial Analysis](#geospatial-analysis)
9. [Results Visualization](#results-visualization)
10. [Contributing](#contributing)

## Project Overview

This project implements a deep learning model to classify bird species based on their audio recordings. It utilizes advanced audio processing techniques, including data augmentation and feature extraction, to build a robust classification model. The project also incorporates geospatial analysis to visualize the distribution of bird species across different locations.

Key features of this project include:
- Audio data augmentation to enhance the dataset
- Comprehensive feature extraction using the librosa library
- A deep learning model based on Convolutional Neural Networks (CNNs)
- Geospatial analysis and visualization of bird species distribution
- Detailed evaluation metrics and result visualization

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bird-audio-classification.git
   cd bird-audio-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the main script:

```bash
python main.py
```

This will perform the following steps:
1. Load and preprocess the data
2. Extract features from the audio files
3. Train the classification model
4. Evaluate the model's performance
5. Generate visualizations of the results

## Data

The dataset incluedes audio recordings that are downloaded from xeno canto.

The dataset consists of audio recordings of various bird species, organized into folders by species. The metadata, including species names and geographical coordinates, is stored in a CSV file named `information.csv`.

## Audio Processing

Audio processing is a crucial step in this project. It includes:

1. Loading audio files using librosa
2. Applying data augmentation techniques
3. Extracting relevant features from the audio signals

## Feature Extraction

Features are extracted from the audio files using the librosa library. The extracted features include:

- Mel-frequency cepstral coefficients (MFCCs)
- Spectral centroid
- Spectral bandwidth
- Chroma features
- Tonnetz

## Model Architecture

The classification model is based on a Convolutional Neural Network (CNN) architecture. It consists of:

- Multiple convolutional layers
- Max pooling layers
- Batch normalization layers
- Dropout for regularization
- Dense layers for classification

## Training Process

The model is trained using the following process:

1. Data splitting into training and validation sets
2. Feature scaling
3. Model compilation with appropriate loss function and optimizer
4. Training with early stopping and learning rate reduction

## Evaluation Metrics

The model's performance is evaluated using several metrics:

- Accuracy
- Confusion matrix
- Per-class precision, recall, and F1-score
- ROC curves and AUC scores

## Geospatial Analysis

Geospatial analysis is performed to visualize the distribution of bird species across different locations. This includes:

- Creating maps using geopandas and folium
- Plotting species distributions
- Analyzing geographical patterns in the data

## Results Visualization

The project includes various visualizations to help interpret the results:

- Training and validation loss/accuracy curves
- Confusion matrix heatmap
- Per-class accuracy bar plots
- Geospatial distribution maps


## Contributing

Contributions to this project are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.
