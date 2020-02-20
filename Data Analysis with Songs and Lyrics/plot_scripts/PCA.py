#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
pca = PCA(0.90) #Forcing into 3 dimensional space

data = pd.read_csv('./vectorizations/vectorized_country.csv')
data.iloc[[2, 4,5,6]].head()


# In[2]:


# indices = []
indices = list(range(4, len(data.columns)))

vectors = data.iloc[:, indices]


# In[3]:


#PCA analysis
from sklearn.decomposition import PCA, KernelPCA
pca = KernelPCA(n_components=3, kernel='rbf')

t_data = pca.fit_transform(vectors)

print(pca)


# In[5]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data.iloc[:, 2], t_data[:, 0], t_data[:, 1], c='orange')

plt.show()


# In[ ]:




