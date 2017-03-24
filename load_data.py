import sys
import numpy as np
from scipy.misc import imread
import dill as pickle
import os
import matplotlib.pyplot as plt
basepath = '/home/soren/Documents/oneshot/omniglot/python/images_background/'
valpath = '/home/soren/Documents/oneshot/omniglot/python/images_evaluation/'
lang_dict = {}
def loadimgs(path,n=0):
	X=[]
	y = []
	cat_dict = {}
	curr_y = n
        #I want to make dict {"lang":(start,end)} for every language
	for alphabet in os.listdir(path):
                lang_dict[alphabet] = [curr_y,None]
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
                lang_dict[alphabet][1] = curr_y - 1
	X=np.stack(X)
	y=np.vstack(y)
#	print(cat_dict)
	return X,y,lang_dict


X,y,c=loadimgs(basepath)

print(lang_dict)
plt.show()
print(X.shape)
print(X.shape)
Xval,yval,cval = loadimgs(valpath)
print(Xval.shape)
with open("train.pickle", "wb") as f:
	pickle.dump((X,y,c),f)


with open("val.pickle", "wb") as f:
	pickle.dump((Xval,yval,cval),f)
