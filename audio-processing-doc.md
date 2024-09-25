# Audio Processing

This document details the audio processing techniques used in the Bird Audio Classification project.

## Overview

Audio processing is a crucial step in preparing the data for the classification model. It involves loading audio files, applying augmentation techniques to increase the dataset's diversity, and extracting relevant features from the audio signals.

## Audio Loading

Audio files are loaded using the librosa library, which provides powerful tools for music and audio analysis.

```python
import librosa

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    return audio, sample_rate
```

The `res_type="kaiser_fast"` parameter is used for efficient resampling of the audio.

## Audio Augmentation

Audio augmentation is applied to increase the diversity of the training data and improve the model's robustness. The following augmentation techniques are implemented:

1. **Time Stretching**
   - Speeds up or slows down the audio without changing its pitch.
   ```python
   def time_stretching(audio):
       return librosa.effects.time_stretch(audio, rate=1.2)
   ```

2. **Pitch Shifting**
   - Shifts the pitch of the audio up or down.
   ```python
   def pitch_shift(audio, sample_rate, n_steps=2):
       return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
   ```

3. **Adding Noise**
   - Adds random Gaussian noise to the audio.
   ```python
   def add_noise(audio, noise_factor=0.005):
       noise = np.random.randn(len(audio))
       return audio + noise_factor * noise
   ```

4. **Random Cropping**
   - Selects a random segment of the audio.
   ```python
   def random_crop(audio, crop_size):
       start = np.random.randint(0, len(audio) - crop_size)
       return audio[start:start + crop_size]
   ```

5. **Volume Change**
   - Increases or decreases the volume of the audio.
   ```python
   def change_volume(audio, factor=1.5):
       return audio * factor
   ```

These augmentation techniques are applied in the `augment_audio` function:

```python
def augment_audio(audio, sample_rate):
    augmented_samples = [
        audio,
        time_stretching(audio),
        pitch_shift(audio, sample_rate, n_steps=2),
        add_noise(audio),
        random_crop(audio, int(sample_rate * 1)),
        change_volume(audio, factor=1.5)
    ]
    return augmented_samples
```

## Considerations

- The augmentation parameters (e.g., pitch shift steps, noise factor) may need to be adjusted based on the specific characteristics of your bird audio dataset.
- Be cautious not to apply too many augmentations, as this might introduce artifacts that could negatively impact the model's performance.
- Consider the computational cost of augmentation, especially for large datasets.

For more details on the feature extraction process that follows audio processing, please refer to `feature_extraction.md`.
