import sys
import numpy as np
from scipy.misc import imread
import dill as pickle
import os
import matplotlib.pyplot as plt
basepath = '/home/soren/Documents/oneshot/omniglot/python/images_background/'
valpath = '/home/soren/Documents/oneshot/omniglot/python/images_evaluation/'

def loadimgs(path,n=0):
	X=[]
	y = []
	cat_dict = {}
	curr_y = n
	for alphabet in os.listdir(path):
		for letter in os.listdir(path+alphabet):
			cat_dict[curr_y] = (alphabet, letter)
			currpath = path + alphabet + '/'
			category_images=[]
			for example in os.listdir(currpath+letter):
				image = imread(currpath+letter+'/'+example)
				category_images.append(image)
				y.append(curr_y)
			X.append(np.stack(category_images))
			curr_y += 1
	X=np.stack(X)
	y=np.vstack(y)
#	print(cat_dict)
	return X,y,cat_dict


X,y,c=loadimgs(basepath)

plt.show()
print(X.shape)
print(X.shape)
Xval,yval,cval = loadimgs(valpath)
print(Xval.shape)
with open("train.pickle", "wb") as f:
	pickle.dump((X,y,c),f)


with open("val.pickle", "wb") as f:
	pickle.dump((Xval,yval,cval),f)
