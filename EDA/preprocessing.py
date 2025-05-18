import os
import h5py
import numpy as np

# Exclude the UAV Controllers from the data
exclude_path = [r"D:\CARDRF\CARDRF\LOS\Train\UAV_Controller", r"D:\CARDRF\CARDRF\LOS\Test\UAV_Controller"]

# Scale factor and segment length
scale_factor = 6.581e-6 
steady_length = 1024

# Initialize lists to hold signals and labels
signals = []
# Bluetooth, UAV, or WiFi
labels = []
# Specific Device Signatures
devices = []

def load_file(filepath):
    with h5py.File(filepath, 'r') as f:
        signal = np.array(f['Channel_1']['Data']).squeeze()
        return signal[3000000:3200000] * scale_factor
    
def detect_steady_region(signal, window=100, threshold_ratio=0.5):
    energy = np.convolve(signal**2, np.ones(window), mode='valid')
    threshold = np.max(energy) * threshold_ratio
    candidates = np.where(energy > threshold)[0]
    if len(candidates) == 0:
        return 0, steady_length
    start = candidates[0]
    end = min(start + steady_length, len(signal))
    return start, end

def iterate_directory(dir_path):
    for filename in os.listdir(dir_path):
        full_path = os.path.join(dir_path, filename)
        
        if full_path in exclude_path:
            continue

        if os.path.isfile(full_path):
            print(f"Processing File: {full_path}")

            signal = load_file(full_path)
            start, end = detect_steady_region(signal)
            segment = signal[start:end]
            signals.append(segment)

            label = os.path.normpath(full_path).split(os.sep)
            labels.append(label[5])
            devices.append(label[6])

        elif os.path.isdir(full_path):
            print(f"Processing New Directory: {full_path}")
            iterate_directory(full_path) 

directory_path = r"D:\CARDRF\CARDRF\LOS"
iterate_directory(directory_path)

signal_out = np.array(signals, dtype=np.float32)
label_out = np.array(labels, dtype=object)
device_out = np.array(devices, dtype=object)

np.save("cardrf_signals.npy", signal_out)
np.save("cardrf_labels.npy", label_out)
np.save("cardrf_devices.npy", device_out)