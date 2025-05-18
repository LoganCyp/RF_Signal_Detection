import h5py
import numpy as np
import matplotlib.pyplot as plt

filepath = r"D:\CARDRF\CARDRF\LOS\Test\BLUETOOTH\APPLE_IPAD3\IPAD00002.mat"  
# scale from digital to voltage as spec'd in paper
scale_factor = 6.581e-6 

# Length of steady state sample (using 1024 as per specs)
steady_length = 1024


with h5py.File(filepath, 'r') as f:
    signal = np.array(f['Channel_1']['Data'])
    signal = signal.squeeze() * scale_factor


window_size=100
threshold_ratio=0.5
    
energy = np.convolve(signal**2, np.ones(window_size), mode='valid')
threshold = np.max(energy) * threshold_ratio
candidates = np.where(energy > threshold)[0]

start = candidates[0]
end = min(start + steady_length, len(signal))

plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.axvline(start, color='green', linestyle='--', label="Steady Start")
plt.axvline(end, color='red', linestyle='--', label="Steady End")
plt.title("DJI Inspire Sample Signal")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(signal[start:end])
plt.title("DJI Inspire Cropped 1024 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()

