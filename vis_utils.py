import sys
import numpy as np
import numpy.random as rng
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
if False:
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

    gradient= [np.linspace(start=0,stop=120,num=img.shape[0])]*img.shape[1]
    gradient = np.asarray(gradient).T
    img += gradient
    plt.matshow(255-img,cmap='gray')
    plt.axis('off')
    plt.show()
class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,Xtrain,Xval=None):
        if Xval is None:
            self.Xval = Xtrain 
        else:
            self.Xval = Xval# / Xval.max()
            self.Xtrain = Xtrain# / Xtrain.max()


        self.n_classes,self.n_examples,self.w,self.h = Xtrain.shape
        self.n_val,self.n_ex_val,_,_ = self.Xval.shape

    def get_batch(self,n):
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(self.n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, self.h, self.w,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0,self.n_examples)
            pairs[0][i,:,:,:] = self.Xtrain[category,idx_1].reshape(self.w,self.h,1)
            idx_2 = rng.randint(0,self.n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else (category + rng.randint(1,self.n_classes)) % self.n_classes
            pairs[1][i,:,:,:] = self.Xtrain[category_2,idx_2].reshape(self.w,self.h,1)
        return pairs, targets

    def make_oneshot_task(self,N):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = rng.choice(self.n_val,size=(N,),replace=False)
        indices = rng.randint(0,self.n_ex_val,size=(N,))
        true_category = categories[0]
        ex1, ex2 = rng.choice(self.n_examples,replace=False,size=(2,))
        test_image = np.asarray([self.Xval[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
        support_set = self.Xval[categories,indices,:,:]
        support_set[0,:,:] = self.Xval[true_category,ex2]
        support_set = support_set.reshape(N,self.w,self.h,1)
        pairs = [test_image,support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets
    
    def test_oneshot(self,model,N,k,verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct

loader = Siamese_Loader(X)
def concat_images(X):
    """Concatenates a bnch of images into a big matrix for plotting purposes."""
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
    return img, n


def plot_oneshot_task(pairs):
    """Takes a one-shot task given to a siamese net and  """
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.set_facecolor('white')
    ax1.set_title("Test Image")
    ax2.set_title("Support Set")
    ax2.grid(linewidth=1,linestyle='-',color='black')
    ax1.matshow(pairs[0][0].reshape(105,105),cmap='gray')
    img,n = concat_images(pairs[1])
    ax2.matshow(img,cmap='gray')
    plt.xticks(np.arange(0,105*n,105))
    plt.yticks(np.arange(0,105*n,105))
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.show()


def format_axis(ax):
    """Adds a grid and removes ticks from ax - designed for plotting grids of digits:w"""
    ax.grid(linewidth-1,linestyle='1',color='black')
    plt.xticks(np.arange(0,105*n,105))
    plt.yticks(np.arange(0,105*n,105))

def plot_alphabet(alphabet_dir,ax=None,title=None):
    """Plots an alphabet given the directory they're stored in. """
    current = os.listdir(alphabet_dir)
    alphabet_length = len(current)
    n = np.ceil(np.sqrt(alphabet_length)).astype("int8")
    x,y=0,0
    w,h = 105,105
    lang_array = np.ones((n*w,n*h))*255
    for l in current:
        letter = os.path.join(alphabet_dir,l)
        first_example = os.listdir(letter)[0]
        first_example = os.path.join(letter,first_example)
        print(l)
        lang_array[x*w:(x+1)*w,y*h:(y+1)*h] = imread(first_example) 
        y += 1
        if y >= n:
            y=0
            x+=1
    if ax is None:
        fig,ax=plt.subplots(1,figsize=(3,3))
    ax.set_title(title)
    ax.matshow(lang_array,cmap='gray')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xticks(np.arange(0,105*n,105))
    plt.yticks(np.arange(0,105*n,105))
    ax.grid(linewidth=1,linestyle='-')
    return fig,ax



base_dir =  "/home/soren/Documents/oneshot/omniglot/python/images_background/"       
save_path = "home/soren/keras-oneshot/"
def save_alphabets(basedir):
    print(len(os.listdir(base_dir)))
    for alphabet in os.listdir(base_dir):
        fig,ax=plot_alphabet(os.path.join(base_dir,alphabet),title=alphabet)
        #filepath = os.path.join(save_path,alphabet)+".png"
        filepath = alphabet + ".png"
        plt.savefig(filepath)
#save_alphabets(base_dir)
#TODO: save all of these a pngs? concat subsets into big images? want them nicely presented
#pairs, _ = loader.make_oneshot_task(9)
#plot_oneshot_task(pairs)

plot_alphabet(os.path.join(base_dir,"Bengali"))
