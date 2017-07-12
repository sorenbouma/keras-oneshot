import sys
import numpy as np
from scipy.misc import imread
import dill as pickle
import os
import matplotlib.pyplot as plt
import argparse
"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""

parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Path where you've downloaded the omniglot images")
args = parser.parse_args()
data_path = args.path
basepath = os.path.join(data_path,'python/images_background')
valpath = os.path.join(data_path,'python/images_evaluation')
lang_dict = {}

def loadimgs(path,n=0):
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    #we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        #every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            #edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict

X,y,c=loadimgs(basepath)


with open("train.pickle", "wb") as f:
	pickle.dump((X,c),f)


X,y,c=loadimgs(basepath)
with open("val.pickle", "wb") as f:
	pickle.dump((X,c),f)
