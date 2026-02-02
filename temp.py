import numpy as np

path="/scratch/zf281/tessera-interactive-map/classification_result_probabilities.npy"
labels = np.load(path)
print(labels.shape)
print(labels[500:520, 500:520,...])