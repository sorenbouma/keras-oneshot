import sys
import numpy as np
from scipy.misc import imread
import dill as pickle
import os
import matplotlib.pyplot as plt
with open("train.pickle", "r") as f:
	(X,y,c) = pickle.load(f)
n_classes,n_ex,h,w=X.shape
s=3
k=8*s
m=19*s

img=[]
for j in range(m):
    imgmatrix = []
    for i in range(k):
        c=np.random.randint(0,n_classes)
        e=np.random.randint(0,n_ex)
        imgmatrix.append(X[c,e])
    imgmatrix=np.vstack(imgmatrix)
    img.append(imgmatrix)
img=np.hstack(img).astype("float64")

gradient= [np.linspace(start=255,stop=0,num=img.shape[0])]*img.shape[1]
gradient = np.asarray(gradient).T
img += gradient
plt.matshow(255-img,cmap='gray')
plt.axis('off')
plt.show()



def plot_oneshot_task(pairs):
	fig, (ax1,ax2) = plt.subplots(2)
	
