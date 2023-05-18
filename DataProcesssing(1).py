import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA

file_path = 'F:\WiAR-master\data1/npy/3/2/v3_a2.dat1a.npy'
data = np.load(file_path,allow_pickle=True)
#data.shape

b, a = signal.butter(8, 0.5, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
filtedData = signal.filtfilt(b, a, data,axis=1)  #data为要过滤的信号

#50Hz，sample 长度5s，2s 有效数据，样本shape:30*100
pca = PCA(n_components=10) #降维至30*10
path = "F:\WiAR-master\data1/npy/3/2/" #存放的目录
data_type = "walk" #动作类型， 
for i in range(data.shape[1]//50):
    data_5s = filtedData[:,5*50*i:5*50*(i+1)]
    #data_5s = data[0,5*50*i:5*50*(i+1)]
    data_2s = data_5s[:,50:3*50]
    data_pca = pca.fit_transform(data_2s)
    np.save(path+str(i),data_pca)# 保存成单个样本
    #print(data_pca.shape)
    #plt.plot(data_pca.reshape(-1))
    #plt.show()
