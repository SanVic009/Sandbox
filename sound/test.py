from moviepy import VideoClip, AudioFileClip
import numpy as np
import soundfile as sf

# Create a dummy audio file
samplerate = 44100
t = np.linspace(0., 1., samplerate)
amplitude = np.iinfo(np.int16).max
data = amplitude * np.sin(2. * np.pi * 440. * t)
sf.write('test_audio.wav', data.astype(np.int16), samplerate)

# Create a dummy video clip
def make_frame(t):
    return np.zeros((100, 100, 3), dtype=np.uint8)

video_clip = VideoClip(make_frame, duration=1)

# Load the audio clip
audio_clip = AudioFileClip("test_audio.wav")

# Try to set the audio
try:
    video_with_audio = video_clip.set_audio(audio_clip)
    print("set_audio worked!")
    video_with_audio.write_videofile("test_video.mp4", fps=24)
    print("Video written successfully!")
except Exception as e:
    print(f"Error: {e}")
