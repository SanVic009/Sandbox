import sounddevice as sd
import numpy as np
import time
import sys

def print_device_info():
    """Print information about audio devices"""
    print("\nAudio Device Information:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:  # Only show input devices
            print(f"\nDevice {i}: {dev['name']}")
            print(f"  Channels (in/out): {dev['max_input_channels']}/{dev['max_output_channels']}")
            print(f"  Sample rate: {dev['default_samplerate']} Hz")
            print(f"  Device API: {dev['hostapi']}")

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)
    volume_norm = np.linalg.norm(indata) * 10
    print(f"Volume: {'#' * int(volume_norm)}")
    if np.any(indata):  # Only print stats if we have non-zero data
        print(f"Stats - Min: {indata.min():.6f}, Max: {indata.max():.6f}, RMS: {np.sqrt(np.mean(indata**2)):.6f}")
    else:
        print("Warning: Receiving zero audio data")
    sys.stdout.flush()  # Ensure output is displayed immediately

print("\nScanning audio devices...")

print_device_info()

# Try each input device until we find one that works
devices = sd.query_devices()
for device_id in range(len(devices)):
    device = devices[device_id]
    if device['max_input_channels'] > 0:
        try:
            print(f"\nTesting device {device_id}: {device['name']}")
            with sd.InputStream(device=device_id,
                              channels=1,
                              callback=audio_callback,
                              blocksize=1024,
                              samplerate=int(device['default_samplerate'])):
                print(f"Recording from {device['name']}... Speak into the microphone.")
                print("Press Ctrl+C to try next device or stop.")
                print("-" * 60)
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print('\nMoving to next device...')
            continue
        except Exception as e:
            print(f'Error with device {device_id}: {str(e)}')
            continue

print("\nNo more devices to test.")
