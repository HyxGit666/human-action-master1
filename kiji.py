import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA

path='F:\WiAR-master\data1\zuizhong/3/1/v3_a21b28.npy'
data_pca= np.load(path,allow_pickle=True)
plt.plot(data_pca.reshape(-1))
plt.show()