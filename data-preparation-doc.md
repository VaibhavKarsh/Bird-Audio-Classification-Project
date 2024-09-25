# Data Preparation

This document describes the data preparation process for the Bird Audio Classification project.

## Dataset Overview

The dataset used in this project consists of audio recordings of various bird species. The data is organized as follows:

- Audio files are stored in folders named `sample{fold}`, where `{fold}` is a number representing the data fold.
- Metadata is stored in a CSV file named `information.csv`.

## Metadata Structure

The `information.csv` file contains the following columns:

- `filename`: Name of the audio file
- `fold`: The fold number for cross-validation
- `name`: Bird species name
- `lat`: Latitude of the recording location
- `lng`: Longitude of the recording location
- `country`: Country where the recording was made

## Data Preparation Steps

1. **Loading the Metadata**
   - The metadata is loaded from `information.csv` using pandas.
   - A sample of the data is taken to ensure reproducibility and manage computational resources.

2. **Data Validation**
   - Check for duplicate entries in the metadata.
   - Identify and handle any missing values.

3. **File Path Construction**
   - Construct the full file path for each audio file using the `fold` and `filename` columns.

4. **Feature Extraction**
   - For each audio file, extract features using the `features_extractor` function.
   - This process is parallelized using `ThreadPoolExecutor` to improve efficiency.

5. **Data Augmentation**
   - During feature extraction, each audio file is augmented to create additional training samples.
   - See `audio_augmentation.md` for details on the augmentation techniques used.

6. **Feature Aggregation**
   - The extracted features from all audio files (including augmented versions) are aggregated into a single dataset.

7. **Data Shuffling**
   - The final dataset is shuffled to ensure random distribution of samples during model training.

## Usage

The data preparation process is implemented in `src/data_preparation.py`. The main function to use is `process_metadata()`, which takes the metadata DataFrame as input and returns a list of extracted features along with their corresponding labels.

Example usage:

```python
from src.data_preparation import process_metadata

metadata = pd.read_csv("information.csv")
extracted_features = process_metadata(metadata)
```

## Considerations

- Ensure that the audio files are present in the correct directories before running the data preparation process.
- The process can be time-consuming for large datasets. Consider using a subset of the data for initial experimentation.
- Monitor system resources during the feature extraction process, as it can be computationally intensive.

For any issues or unexpected behavior during the data preparation process, please refer to the troubleshooting section in `usage.md` or open an issue on the project's GitHub repository.
