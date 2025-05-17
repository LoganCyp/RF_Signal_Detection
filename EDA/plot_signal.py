import h5py
import numpy as np
import matplotlib.pyplot as plt

filepath = r"D:\CARDRF\CARDRF\LOS\Test\UAV\BEEBEERUN\FLYING\BEEBEERUN_0000100002.mat"  
# scale from digital to voltage as spec'd in paper
scale_factor = 6.581e-6 


with h5py.File(filepath, 'r') as f:
    signal = np.array(f['Channel_1']['Data'])
    signal = signal.squeeze() * scale_factor



plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title("Beebeerun Sample Signal")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()
