# Load library
import numpy as np
from findpeaks import findpeaks

# Data
i = 10000
xs = np.linspace(0,3.7*np.pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Initialize
fp = findpeaks(method='peakdetect')
results = fp.fit(X)
# Plot
fp.plot1d()

fp = findpeaks(method='topology', limit=1)
results = fp.fit(X)
fp.plot1d()
fp.plot_persistence()
