import numpy as np
labels = np.load('output/labels.npy')

# Check unique classes
print("Labels shape:", labels.shape)
print("Unique classes:", np.unique(labels))