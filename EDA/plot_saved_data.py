import numpy as np
import matplotlib.pyplot as plt

# Load saved files
X = np.load("cardrf_signals.npy")
y = np.load("cardrf_labels.npy", allow_pickle=True)  

y_device = np.load("cardrf_devices.npy", allow_pickle=True)

print(f"X Shape: {len(X)}")
print(f"Y Shape: {y.shape}")

print(np.unique(y_device))

# Extract the signal and label
signal = X[100]
label = y[100]
device = y_device[100]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title(f"Label: {label} \n Device: {device}")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()
