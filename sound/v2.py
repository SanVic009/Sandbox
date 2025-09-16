import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import soundfile as sf

# --------- Config ---------
AUDIO_FILE = "audio.wav"
N_SPEAKERS = 2
FRAME_SIZE = 2.0   # seconds
FRAME_STEP = 1.0   # seconds
N_MFCC = 13
SR = 16000

# --------- Load Audio ---------
signal, sr = librosa.load(AUDIO_FILE, sr=SR)

# --------- Extract MFCC Features ---------
def extract_features(signal, sr):
    mfccs = []
    timestamps = []
    hop_length = int(FRAME_STEP * sr)
    win_length = int(FRAME_SIZE * sr)
    
    for start in range(0, len(signal) - win_length, hop_length):
        end = start + win_length
        frame = signal[start:end]
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfccs.append(mfcc_mean)
        timestamps.append(start / sr)
    
    return np.array(mfccs), np.array(timestamps)

features, timestamps = extract_features(signal, sr)

# --------- Normalize ---------
scaler = StandardScaler()
X = scaler.fit_transform(features)

# --------- Train GMM ---------
gmm = GaussianMixture(n_components=N_SPEAKERS, covariance_type='diag', random_state=0)
gmm.fit(X)
labels = gmm.predict(X)

# --------- Plot Results ---------
plt.figure(figsize=(14, 4))
plt.plot(np.arange(len(signal)) / sr, signal, alpha=0.5)
for i, t in enumerate(timestamps):
    plt.axvspan(t, t + FRAME_SIZE, color=f"C{labels[i]}", alpha=0.3)
plt.title("Speaker Diarization using MFCC + GMM")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
