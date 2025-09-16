# live_diarize.py
import queue
import threading
import time
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
import librosa
import sys
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import webrtcvad
import matplotlib.pyplot as plt

# ---------- Config ----------
import sounddevice as sd

def get_supported_samplerate():
    device_info = sd.query_devices(kind='input')
    return int(device_info['default_samplerate'])

PROCESSING_SR = 16000      # Sample rate for processing
DEVICE_SR = get_supported_samplerate()  # Use the device's default sample rate
BLOCK_SEC = 0.1            # Reduced block size for lower latency
FEAT_WIN_SEC = 0.5         # Reduced feature window length
FEAT_HOP_SEC = 0.2         # Reduced hop size for more frequent updates
BUFFER_SIZE = 1024         # Audio buffer size
N_MFCC = 13
MAX_BUFFER_SEC = 300       # Keep last N seconds in memory for model refit
REFIT_PERIOD_SEC = 3.0     # Refit clustering this often
MAX_INIT_COMPONENTS = 12   # Upper cap for DP-GMM; effective components inferred
VAD_AGGRESSIVENESS = 1     # 0-3; bigger -> more aggressive (using less aggressive setting)
VAD_FRAME_MS = 30          # webrtcvad supported: 10/20/30
SPEECH_RATIO_THRESH = 0.5  # fraction of subframes that must be speech to accept a block
MIN_SEGMENT_SEC = 0.7      # minimum segment duration to emit
PLOT = True                # live plot of labels timeline

# ---------- Utilities ----------
def secs_to_hhmmss(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

@dataclass
class FrameFeat:
    t_start: float
    t_end: float
    x: np.ndarray
    label: int|None = None
    conf: float|None = None  # softmax-ish cluster posterior max

class LabelStabilizer:
    """
    Keep speaker IDs stable across successive model refits by matching
    new cluster means to previous ones via cosine similarity.
    """
    def __init__(self):
        self.prev_means = None
        self.permutation = None  # maps new -> stable
        self.next_id = 0
        self.id_map = {}  # new_index -> stable_id

    @staticmethod
    def _cosine_sim(a, b, eps=1e-9):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True)+eps)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True)+eps)
        return a @ b.T  # (k_new x k_old)

    def update(self, new_means, new_weights, weight_thresh=1e-3):
        # Filter out near-empty components
        keep = new_weights > weight_thresh
        new_means = new_means[keep]
        idx_map = np.nonzero(keep)[0]
        new_to_stable = {}

        if self.prev_means is None or len(self.prev_means)==0 or len(new_means)==0:
            # Assign fresh stable ids
            for i, _ in enumerate(new_means):
                sid = self.next_id; self.next_id += 1
                new_to_stable[idx_map[i]] = sid
            self.prev_means = new_means.copy()
            self.id_map = new_to_stable.copy()
            return self.id_map

        S = self._cosine_sim(new_means, self.prev_means)
        # Greedy matching
        used_old = set()
        for i in np.argsort(-S.max(axis=1)):  # order by best match strength
            j = int(np.argmax(S[i]))
            if j not in used_old:
                # Reuse old id
                stable_j = self.id_map.get(j, None)
                if stable_j is None:
                    stable_j = self.next_id; self.next_id += 1
                new_to_stable[idx_map[i]] = stable_j
                used_old.add(j)
        # Unmatched -> new ids
        for i in range(len(new_means)):
            if idx_map[i] not in new_to_stable:
                sid = self.next_id; self.next_id += 1
                new_to_stable[idx_map[i]] = sid

        # Update prev means to new means in stable order (approximate)
        self.prev_means = new_means.copy()
        self.id_map = new_to_stable.copy()
        return self.id_map

class LiveDiarizer:
    def __init__(self):
        self.audio_q = queue.Queue()
        self.stop_event = threading.Event()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.block_samples = int(BLOCK_SEC * DEVICE_SR)
        self.win_samples = int(FEAT_WIN_SEC * PROCESSING_SR)
        self.hop_samples = int(FEAT_HOP_SEC * PROCESSING_SR)
        self.ring = np.zeros(0, dtype=np.float32)
        self.t0 = None
        self.features: list[FrameFeat] = []
        self.scaler = StandardScaler()
        self.model = None
        self.last_refit_t = 0.0
        self.lab_stab = LabelStabilizer()
        self.current_seg_label = None
        self.current_seg_start = None
        self.current_seg_conf_acc = 0.0
        self.current_seg_frames = 0
        self.resampler = None  # Will be initialized when needed

        if PLOT:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 3))
            self.line = None

    # ---------- Audio ----------
    def _callback(self, indata, frames, time_info, status):
        try:
            if status:
                # Just log overflow errors instead of printing each one
                if 'overflow' in str(status):
                    return
                print(f"Audio callback status: {status}")
                return
            
            # If we have stereo, average the channels
            if indata.shape[1] == 2:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata[:, 0]
            
            # Apply gain to boost the signal
            mono = mono * 5.0  # Boost the signal
            
            # Check if we're getting audio data
            rms = np.sqrt(np.mean(mono**2))
            if rms > 0.01:  # Only print when there's significant audio
                print(f"Audio input RMS: {rms:.6f}")
                
            # Add to queue
            self.audio_q.put(mono.copy())
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            import traceback
            traceback.print_exc()
            return

    def start_stream(self):
        try:
            # Get device info
            device_info = sd.query_devices(mic_device)
            channels = min(2, device_info['max_input_channels'])  # Use up to 2 channels
            
            stream = sd.InputStream(
                samplerate=DEVICE_SR,
                device=mic_device,  # Use our found working microphone
                channels=channels,
                dtype='float32',
                blocksize=BUFFER_SIZE,
                callback=self._callback,
                latency='low'
            )
            print(f"\nOpening stream with:")
            print(f"- Device: {device_info['name']}")
            print(f"- Channels: {channels}")
            print(f"- Sample rate: {DEVICE_SR}")
            
            stream.start()
            return stream
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise

    def _resample(self, audio):
        return librosa.resample(audio, orig_sr=DEVICE_SR, target_sr=PROCESSING_SR)

    # ---------- VAD ----------
    def _is_speech_block(self, block):
        # Print audio block statistics
        rms = np.sqrt(np.mean(block**2))
        print(f"\nAnalyzing audio block - RMS: {rms:.6f}, Max: {np.max(np.abs(block)):.6f}")
        
        # Resample to processing rate for VAD
        resampled = self._resample(block)
        print(f"Resampled length: {len(resampled)}, Original length: {len(block)}")
        
        # webrtcvad expects 16-bit PCM, 10/20/30 ms chunks
        step = int(PROCESSING_SR * (VAD_FRAME_MS / 1000.0))
        pcm16 = np.clip(resampled, -1, 1)
        pcm16 = (pcm16 * 32768.0).astype(np.int16).tobytes()

        speech_count = 0
        total = 0
        for i in range(0, len(block) - step + 1, step):
            frame = pcm16[i*2:(i+step)*2]  # 2 bytes per sample
            if len(frame) != step*2:
                break
            if self.vad.is_speech(frame, PROCESSING_SR):
                speech_count += 1
            total += 1
        if total == 0:
            return False, 0.0
        ratio = speech_count / total
        return ratio >= SPEECH_RATIO_THRESH, ratio

    # ---------- Features ----------
    def _extract_feat_windowed(self, buf, t_start):
        feats = []
        times = []
        
        # Print debug info about the buffer
        print(f"Buffer length: {len(buf)}, Window samples: {self.win_samples}, Hop samples: {self.hop_samples}")
        
        for s in range(0, len(buf) - self.win_samples + 1, self.hop_samples):
            win = buf[s:s+self.win_samples]
            
            # Ensure we have enough samples
            if len(win) == self.win_samples:
                try:
                    mfcc = librosa.feature.mfcc(y=win, sr=PROCESSING_SR, n_mfcc=N_MFCC)
                    mfcc_mean = np.mean(mfcc, axis=1)
                    feats.append(mfcc_mean)
                    times.append(t_start + s / PROCESSING_SR)
                except Exception as e:
                    print(f"Error extracting MFCC features: {e}")
                    continue
                    
        # Print debug info about extracted features
        print(f"Extracted {len(feats)} feature vectors")
        return np.array(feats), np.array(times)

    # ---------- Clustering ----------
    def _refit(self):
        if len(self.features) < 4:
            print("Not enough features for clustering yet")
            return
            
        print("\n--- Clustering Update ---")
        print(f"Processing {len(self.features)} features")
        
        X = np.stack([f.x for f in self.features], axis=0)
        print(f"Feature matrix shape: {X.shape}")
        
        # Fit scaler on current buffer
        Xn = self.scaler.fit_transform(X)
        
        # Bayesian GMM (Dirichlet Process)
        bgmm = BayesianGaussianMixture(
            n_components=MAX_INIT_COMPONENTS,
            covariance_type='diag',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.3,   # lower -> fewer active clusters
            max_iter=300,
            random_state=0,
            init_params='kmeans',
            reg_covar=1e-5
        )
        
        bgmm.fit(Xn)
        
        # Print clustering results
        active_components = np.sum(bgmm.weights_ > 0.01)
        print(f"Active components detected: {active_components}")
        
        # Stabilize labels
        id_map = self.lab_stab.update(bgmm.means_, bgmm.weights_)
        print(f"Stable speaker IDs: {sorted(set(id_map.values()))}")
        
        resp = bgmm.predict_proba(Xn)
        raw_labels = np.argmax(resp, axis=1)
        conf = resp[np.arange(len(resp)), raw_labels]
        
        print(f"Average confidence: {np.mean(conf):.3f}")
        
        # Map to stable IDs and assign
        for i, (rl, c) in enumerate(zip(raw_labels, conf)):
            if rl in id_map:
                self.features[i].label = id_map[rl]
                self.features[i].conf = float(c)
            else:
                sid = self.lab_stab.next_id
                self.lab_stab.next_id += 1
                self.features[i].label = sid
                self.features[i].conf = float(c)
        self.model = bgmm

    # ---------- Segmentation + Output ----------
    def _emit_if_needed(self, new_label, new_conf, t_end):
        # Start new
        if self.current_seg_label is None:
            self.current_seg_label = new_label
            self.current_seg_start = t_end - FEAT_WIN_SEC
            self.current_seg_conf_acc = new_conf
            self.current_seg_frames = 1
            return
        # Same label -> extend
        if new_label == self.current_seg_label:
            self.current_seg_conf_acc += new_conf
            self.current_seg_frames += 1
            return
        # Label changed -> possibly emit
        seg_dur = t_end - self.current_seg_start
        if seg_dur >= MIN_SEGMENT_SEC:
            avg_conf = self.current_seg_conf_acc / max(1, self.current_seg_frames)
            print(f"[{secs_to_hhmmss(self.current_seg_start)}–{secs_to_hhmmss(t_end)}] "
                  f"Speaker S{self.current_seg_label} (p={avg_conf:.2f})")
        # Start next
        self.current_seg_label = new_label
        self.current_seg_start = t_end - FEAT_WIN_SEC
        self.current_seg_conf_acc = new_conf
        self.current_seg_frames = 1

    def _trim_buffer(self):
        # Keep only last MAX_BUFFER_SEC
        cutoff = (self.t_now() - MAX_BUFFER_SEC)
        keep = [f for f in self.features if f.t_end >= cutoff]
        self.features = keep

    def t_now(self):
        return time.time() - self.t0

    # ---------- Plot ----------
    def _update_plot(self):
        if not PLOT:
            return
            
        if len(self.features) == 0:
            return
            
        try:
            t = [f.t_start for f in self.features]
            y = [f.label if f.label is not None else -1 for f in self.features]
            
            self.ax.clear()
            self.ax.scatter(t, y, s=18, alpha=0.6)
            
            # Set reasonable axis limits
            if len(t) > 0:
                t_min, t_max = min(t), max(t)
                self.ax.set_xlim(max(0, t_min), t_max + 1)
                
                if any(label != -1 for label in y):
                    self.ax.set_ylim(-0.5, max(y) + 0.5)
                else:
                    self.ax.set_ylim(-0.5, 0.5)
            
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Speaker ID")
            self.ax.set_title("Live Speaker Diarization")
            self.ax.grid(True, alpha=0.3)
            
            # Add some padding around the plot
            plt.tight_layout()
            
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception as e:
                # Ignore visualization errors
                pass
                
        except Exception as e:
            # Ignore visualization errors
            pass

    # ---------- Main loop ----------
    def run(self):
        self.t0 = time.time()
        print("Initializing audio stream...", flush=True)
        stream = self.start_stream()
        print(f"Audio configuration:")
        print(f"- Device sample rate: {DEVICE_SR} Hz")
        print(f"- Processing sample rate: {PROCESSING_SR} Hz")
        print(f"- Buffer size: {BUFFER_SIZE} samples")
        print(f"- Block duration: {BLOCK_SEC*1000:.1f} ms")
        print("\nListening… Ctrl+C to stop.", flush=True)

        try:
            buf = np.zeros(0, dtype=np.float32)
            last_refit = 0.0
            next_feat_cut = 0
            last_overflow_warning = 0

            while not self.stop_event.is_set():
                try:
                    block = self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                t_block_start = self.t_now()
                # VAD gate
                is_speech, ratio = self._is_speech_block(block)
                if not is_speech:
                    continue

                # Append to buffer and process into feature frames
                buf = np.concatenate([buf, block])
                # Resample the entire buffer to processing rate
                resampled_buf = self._resample(buf)
                # Only process when we have at least one hop beyond current cut
                while len(resampled_buf) - next_feat_cut >= self.hop_samples + self.win_samples:
                    s = next_feat_cut
                    e = s + self.win_samples
                    win = resampled_buf[s:e]
                    t_start = self.t_now() - (len(resampled_buf) - s)/PROCESSING_SR
                    mfcc = librosa.feature.mfcc(y=win, sr=PROCESSING_SR, n_mfcc=N_MFCC)
                    mfcc_mean = np.mean(mfcc, axis=1)
                    t_end = t_start + FEAT_WIN_SEC
                    self.features.append(FrameFeat(t_start, t_end, mfcc_mean))
                    next_feat_cut += self.hop_samples

                # Refit periodically
                now = self.t_now()
                if now - last_refit >= REFIT_PERIOD_SEC:
                    self._trim_buffer()
                    self._refit()
                    # Emit rolling segment for the newest labeled frame
                    if len(self.features) and self.features[-1].label is not None:
                        self._emit_if_needed(self.features[-1].label,
                                             self.features[-1].conf or 0.0,
                                             self.features[-1].t_end)
                    self._update_plot()
                    last_refit = now

        except KeyboardInterrupt:
            pass
        finally:
            # Emit final segment if long enough
            if self.current_seg_label is not None:
                seg_end = self.features[-1].t_end if self.features else self.t_now()
                seg_dur = seg_end - self.current_seg_start
                if seg_dur >= MIN_SEGMENT_SEC:
                    avg_conf = self.current_seg_conf_acc / max(1, self.current_seg_frames)
                    print(f"[{secs_to_hhmmss(self.current_seg_start)}–{secs_to_hhmmss(seg_end)}] "
                          f"Speaker S{self.current_seg_label} (p={avg_conf:.2f})")
            stream.stop()
            stream.close()
            if PLOT:
                plt.ioff()
                plt.show()

def find_working_microphone():
    return 8  # Use the device we found that works well
    
def _find_working_microphone():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # This is an input device
            try:
                print(f"\nTesting device {i}: {device['name']}")
                # Try to record a small sample
                with sd.InputStream(device=i, channels=1, dtype='float32', callback=None):
                    print(f"Device {i} is working!")
                    return i
            except Exception as e:
                print(f"Device {i} failed: {e}")
    return None

if __name__ == "__main__":
    try:
        print("Available input devices:")
        print(sd.query_devices())
        
        # Find a working microphone
        mic_device = find_working_microphone()
        if mic_device is None:
            print("No working microphone found!")
            sys.exit(1)
            
        # Get the device info
        device_info = sd.query_devices(mic_device)
        DEVICE_SR = int(device_info['default_samplerate'])
        
        print(f"\nUsing input device {mic_device}: {device_info['name']}")
        print(f"Channels: {device_info['max_input_channels']}")
        print(f"Sample rate: {DEVICE_SR} Hz")
        
        diar = LiveDiarizer()
        diar.run()
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        print("\nDevice capabilities:", file=sys.stderr)
        device_info = sd.query_devices(kind='input')
        print(f"Default input device info:\n{device_info}", file=sys.stderr)
        sys.exit(1)
