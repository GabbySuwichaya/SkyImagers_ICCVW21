import h5py
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
from warp import *
import pdb

import matplotlib.pyplot as plt
## Hyperparameter
WarpUpSampFactor = 0.707
#############

class DatasetFromFolder(data.Dataset):
    def __init__(self, input_files, target_files):
        super(DatasetFromFolder, self).__init__()

        self.inputFile  = h5py.File(input_files, 'r')
        self.targetFile = h5py.File(target_files, 'r') 
        
        self.n_images = len(self.inputFile)

    def __getitem__(self, index):
        XfileName = 'X' + str(index)
        YfileName = 'y' + str(index)
        
        inputs = self.inputFile[XfileName]  
        inputs = inputs[:]
        inputs = np.float32(inputs)
        inputs = inputs/255
        #Warp Here
        inputs = warp(inputs, WarpUpSampFactor)
        inputs = np.moveaxis(inputs, 2, 0)
        inputs = torch.from_numpy(inputs)
        
        
        target = self.targetFile[YfileName]
        target = target[:]
        target = np.float32(target)
        target = target/255
        #Warp Here
        target = warp(target, WarpUpSampFactor)
        
        target = np.moveaxis(target, 2, 0)
        target = torch.from_numpy(target)
        
        
        inputs = np.float32(inputs)
        target = np.float32(target)

        # pdb.set_trace() << Comment อันนี้ ออกด้วยนะคะ 
        return inputs, target

    def __len__(self):
        return self.n_images
    

def plot_patch(ax, X, text): 
    ax.imshow(X)
    ax.set_title(text)
    ax.axis('equal')
    ax.axis('off')


def plotXY_15x18(input_X,input_Y):
    X_ = input_X.permute(2,3,1,0)
    Y_ = input_Y.permute(2,3,1,0)
 
    Y1  = Y_[:,:,:3,0] 
    Y2  = Y_[:,:,3:6,0] 
    Y3  = Y_[:,:,6:9,0] 
    Y4  = Y_[:,:,9:12,0] 
    Y5  = Y_[:,:,12:15,0] 

    X1 = X_[:,:,:3,0] 
    X2 = X_[:,:,3:6,0] 
    X3 = X_[:,:,6:9,0]  
    X4 = X_[:,:,9:12,0]
    X5 = X_[:,:,12:15,0]
    X6 = X_[:,:,15:18,0]

 
    fig, axs = plt.subplots(2,6,figsize=(15, 6)) 
    
    plot_patch(axs[0,0], Y1, "Y1")   
    plot_patch(axs[0,1], Y2, "Y2")  
    plot_patch(axs[0,2], Y3, "Y3")  
    plot_patch(axs[0,3], Y4, "Y4")  
    plot_patch(axs[0,4], Y5, "Y5")  


    plot_patch(axs[1,0], X1, "X1")  
    plot_patch(axs[1,1], X2, "X2")
    plot_patch(axs[1,2], X3, "X3")
    plot_patch(axs[1,3], X4, "X4")
    plot_patch(axs[1,4], X5, "X5")
    plot_patch(axs[1,5], X6, "X6")

    plt.show()    


def plotXY(input_X,input_Y, Y_predict=None, savepath=None, text_description=None):
    
 
    Y  = input_Y[:,:,:3] 

    X1 = input_X[:,:,:3] 
    X2 = input_X[:,:,3:6] 
    X3 = input_X[:,:,6:9]  
    X4 = input_X[:,:,9:12]

    # [ 'Sampled data:', 
    #                               "Empirical Mean: %.2f"  % np.mean(data_sample), 
    #                               "Empirical std: %.2f" % np.std(data_sample)]

    if text_description is not None:    
        txt_list = []
        for key, value in text_description.items():
            if key == "id":
                txt_list.append( "%s: %d" % (key, value))
            else:
                txt_list.append( "%s: %.2f" % (key, value))
    
        textstr_sampled = '\n'.join(txt_list)   

    if Y_predict is None:
        fig, axs = plt.subplots(1,5,figsize=(15, 3))  

        plot_patch(axs[0], X1, "X1")  
        if text_description is not None:
            axs[0].text(-0, 1.5,   textstr_sampled,   horizontalalignment='left',    verticalalignment='top',  family='monospace', color="blue") 
        plot_patch(axs[1], X2, "X2")
        plot_patch(axs[2], X3, "X3")
        plot_patch(axs[3], X4, "X4")
        plot_patch(axs[4], Y, "Y")   
    else: 
        fig, axs = plt.subplots(1,6,figsize=(15, 3))  
        plot_patch(axs[0], X1, "X1")  
        if text_description is not None:
            axs[0].text(-0, 280,   textstr_sampled,  horizontalalignment='left',    verticalalignment='top',  family='monospace', color="blue") 
        plot_patch(axs[1], X2, "X2")
        plot_patch(axs[2], X3, "X3")
        plot_patch(axs[3], X4, "X4")
        plot_patch(axs[4], Y,  "Y GT")    
        plot_patch(axs[5], Y_predict, "Y Predict")             

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)    


if __name__ == "__main__":

    INPUTS_PATH = "./SkyNet_Data/xTest_skip_pred5.h5"
    TARGET_PATH = "./SkyNet_Data/yTest_skip_pred5.h5"
    dataset = DatasetFromFolder(INPUTS_PATH, TARGET_PATH)
    dataloader = DataLoader(dataset)
    for data_ in dataloader:
        inputs, target = data_  
        long_inputs = torch.cat([inputs,target],dim=1)

        for i in range(5): 
            x = long_inputs[:,3*i:(3*i+12),:,:] 
            y = target[:,3*i:(3*i+3),:,:]   
            plotXY(x.permute(2,3,1,0).squeeze(-1), y.permute(2,3,1,0).squeeze(-1))  
            




  
