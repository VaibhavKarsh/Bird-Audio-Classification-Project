import librosa
import numpy as np
from src.audio_augmentation import augment_audio

def features_extractor(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        
        augmented_audios = augment_audio(audio, sample_rate)

        features_list = []
        for augmented_audio in augmented_audios:
            mfccs = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=40)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            spectral_centroid = librosa.feature.spectral_centroid(y=augmented_audio, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=augmented_audio, sr=sample_rate, roll_percent=0.85)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=augmented_audio, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=augmented_audio, sr=sample_rate)
            chroma = librosa.feature.chroma_stft(y=augmented_audio, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=augmented_audio)
            tonnetz = librosa.feature.tonnetz(y=augmented_audio, sr=sample_rate)

            features = np.concatenate([
                np.mean(mfccs.T, axis=0),
                np.mean(mfccs_delta.T, axis=0),
                np.mean(mfccs_delta2.T, axis=0),
                np.mean(spectral_centroid.T, axis=0),
                np.mean(spectral_rolloff.T, axis=0),
                np.mean(spectral_bandwidth.T, axis=0),
                np.mean(spectral_contrast.T, axis=0),
                np.mean(chroma.T, axis=0),
                np.mean(zero_crossing_rate.T, axis=0),
                np.mean(tonnetz.T, axis=0)
            ])
            
            features = (features - np.mean(features)) / (np.std(features) + 1e-7)
            features_list.append(features)
        
        return features_list

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None
