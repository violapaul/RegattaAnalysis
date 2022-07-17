

# %%

import numpy as np
import matplotlib.pyplot as plt
# %%


size = 16
x = np.linspace(0, np.pi * 2, size)
y = np.linspace(0, np.pi * 2, size)

xx, yy = np.meshgrid(x, y)

print(xx)
print(yy)

plt.imshow(xx)


# %%
sig = np.sin(xx * 2) * np.sin(yy * 3)
# sig = 1.0 * np.sin(yy * 2)

plt.imshow(sig)

# %%

sig = np.zeros((128, 128))
sig[0:10, 0:40] = 1

plt.imshow(sig)

# %%

fsig = np.fft.fft2(sig)
fsig = np.fft.ifftshift(fsig)
plt.imshow(np.abs(fsig))


# %%

isig = np.fft.ifft2(sig)
plt.imshow(np.abs(isig))
# %%

print(fsig.max(), fsig.min())

# %%

sig = np.zeros((128))
sig[0:30] = 1
plt.plot(sig)

# %%

fsig = np.fft.fft(sig)
fsig = np.fft.fftshift(fsig)
plt.plot(np.abs(fsig))

# %%

isig = np.fft.ifft(sig)
isig = np.fft.fftshift(isig)
plt.plot(np.abs(isig))