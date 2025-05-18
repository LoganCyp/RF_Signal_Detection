import numpy as np
import matplotlib.pyplot as plt

# Load saved files
X = np.load("cardrf_signals.npy")         # shape: (N, 1024)
y = np.load("cardrf_labels.npy", allow_pickle=True)  # shape: (N,)

print(len(X))
print(y.shape)

# Extract the signal and label
signal = X[5801]
label = y[5801]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title(f"Label: {label}")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()
