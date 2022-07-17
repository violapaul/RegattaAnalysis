
import numpy as np

import math

print(2300000000 * 0.0038)



fig, ax = plt.subplots(figsize=(12, 6))

num_days = 7

xx = np.linspace(0, num_days)

yy = np.power(0.85,  num_days - xx)

ax.clear()
ax.plot(xx, yy)
ax.set_ylim((0, 1.1))


