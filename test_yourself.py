import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from util import Siamese_Loader

with open("train.pickle","r") as f:
    (Xtrain,y,c) = pickle.load(f)

with open("val.pickle","r") as f:
    (Xval,y,c) = pickle.load(f)

def concat_images(X):
    nc,h,w,_ = X.shape
    X = X.reshape(nc,h,w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img

def plot_oneshot_task(pairs):
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(105,105),cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
loader = Siamese_Loader(Xtrain,Xval)
pairs, targets = loader.make_oneshot_task(22)

def test_human_oneshot(N_ways,loader,n_trials):
    for trial in range(n_trials):
        inputs, targets = loader.make_oneshot_task(N_ways)
        plot_oneshot_task(inputs)
        plt.close()
test_human_oneshot(5,loader,100)


