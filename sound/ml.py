import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from moviepy import AudioFileClip, VideoClip
import soundfile as sf

# ----- Config -----
AUDIO_FILE = "audio.wav"
SR = 16000
FRAME_SIZE = 2.0  # in seconds
FRAME_STEP = 1.0  # in seconds
N_SPEAKERS = 2
N_MFCC = 13
VIDEO_OUT = "diarization.mp4"
FPS = 24  # Increased for smoother playback

# ----- Load audio -----
try:
    y, sr = librosa.load(AUDIO_FILE, sr=SR)
except Exception as e:
    print(f"Error loading audio file: {e}")
    exit(1)

duration = len(y) / sr
print(f"Audio duration: {duration:.2f} seconds")

# ----- Feature Extraction -----
def extract_features(y, sr):
    win_length = int(FRAME_SIZE * sr)
    hop_length = int(FRAME_STEP * sr)

    features = []
    timestamps = []

    for start in range(0, len(y) - win_length, hop_length):
        end = start + win_length
        frame = y[start:end]
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.append(mfcc_mean)
        timestamps.append(start / sr)

    return np.array(features), np.array(timestamps)

features, timestamps = extract_features(y, sr)
print(f"Extracted {len(features)} feature frames")

def plot_mfcc(features, timestamps):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(features.T, x_axis='time', sr=sr, hop_length=int(FRAME_STEP * sr))
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

plot_mfcc(features, timestamps)

# ----- Normalize and Cluster -----
scaler = StandardScaler()
X = scaler.fit_transform(features)

gmm = GaussianMixture(n_components=N_SPEAKERS, random_state=42)
labels = gmm.fit_predict(X)
print(f"Clustered into {N_SPEAKERS} speakers")

# ----- Create Frame for Video -----
def make_frame(t):
    fig, ax = plt.subplots(figsize=(10, 2), dpi=100)  # 1000x200 pixels

    # Plot only the visible part of the waveform to save memory
    start_time_window = max(0, t - 2)
    end_time_window = min(duration, t + 2)
    start_sample = int(start_time_window * sr)
    end_sample = int(end_time_window * sr)

    visible_y = y[start_sample:end_sample]
    if len(visible_y) > 0:
        visible_t = np.linspace(start_time_window, end_time_window, len(visible_y))
        ax.plot(visible_t, visible_y, color='lightgray', linewidth=0.8)

    # Overlay active speaker segment at time t
    for i, start_time in enumerate(timestamps):
        if start_time <= t < start_time + FRAME_SIZE:
            ax.axvspan(start_time, start_time + FRAME_SIZE, color=f"C{labels[i]}", alpha=0.4)

    ax.set_xlim(max(0, t - 2), min(duration, t + 2))
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Speaker Diarization")
    plt.tight_layout()

    # Convert to RGB image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Include alpha channel
    img = img[:, :, :3]  # Drop alpha channel for RGB
    plt.close(fig)
    return img

# ----- Generate Video -----
try:
    audio_clip = AudioFileClip(AUDIO_FILE)
    video_clip = VideoClip(make_frame, duration=duration)
    video_clip.audio = audio_clip
    video_clip.write_videofile(VIDEO_OUT, fps=FPS, codec='libx264', audio_codec='aac')
    audio_clip.close()
    video_clip.close()
    print(f"Video saved as {VIDEO_OUT}")
except Exception as e:
    print(f"Error generating video: {e}")