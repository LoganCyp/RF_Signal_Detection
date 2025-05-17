import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

root_dir = r"D:\CARDRF\CARDRF\LOS"
scale_factor = 6.581e-6
steady_length = 1024

def find_first_mat_file(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".mat"):
                return os.path.join(root, file)
    return None

def load_signal(filepath):
    with h5py.File(filepath, 'r') as f:
        signal = np.array(f['Channel_1']['Data']).squeeze()
        return signal * scale_factor

def detect_steady_region(signal, window_size=100, threshold_ratio=0.5):
    energy = np.convolve(signal**2, np.ones(window_size), mode='valid')
    threshold = np.max(energy) * threshold_ratio
    candidates = np.where(energy > threshold)[0]
    if len(candidates) == 0:
        return 0, steady_length  # fallback
    start = candidates[0]
    end = min(start + steady_length, len(signal))
    return start, end

def plot_signal_with_crop(signal, start, end, title):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="Full Signal")
    plt.axvline(start, color='green', linestyle='--', label="Steady Start")
    plt.axvline(end, color='red', linestyle='--', label="Steady End")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cropped_signal(signal, title):
    plt.figure(figsize=(10, 3))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Walk through folders ---
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if not os.path.isdir(category_path): continue

    for sub in os.listdir(category_path):
        sub_path = os.path.join(category_path, sub)
        if not os.path.isdir(sub_path): continue

        mat_file = find_first_mat_file(sub_path)
        if not mat_file: continue

        signal = load_signal(mat_file)
        start, end = detect_steady_region(signal)

        title = f"{category}/{sub}: {os.path.basename(mat_file)}"
        plot_signal_with_crop(signal, start, end, title)
        plot_cropped_signal(signal[start:end], f"Steady-State Crop: {title}")

