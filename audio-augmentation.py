import librosa
import numpy as np

def time_stretching(audio):
    return librosa.effects.time_stretch(audio, rate=1.2)

def pitch_shift(audio, sample_rate, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented

def random_crop(audio, crop_size):
    start = np.random.randint(0, len(audio) - crop_size)
    return audio[start:start + crop_size]

def change_volume(audio, factor=1.5):
    return audio * factor

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
