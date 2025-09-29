import numpy as np

# Load the features
features = np.load('output/features.npy')

# Check shape and content
print("Features shape:", features.shape)  # (n_samples, n_features)
print("First 5 features:\n", features[:5])