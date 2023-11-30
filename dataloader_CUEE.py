import h5py
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np
import torch
from warp import *
import pdb
import glob
from math import log10, sqrt 
import matplotlib.pyplot as plt
import random

## Hyperparameter
WarpUpSampFactor = 0.707

def PSNR_torch(original, compressed):  
    B, C, H, W = original.shape
    mse = torch.mean((original.view(B, C, -1) - compressed.view(B, C, -1)) ** 2, dim=2)   
    max_pixel = 255 
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))  
    return psnr

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 


class DatasetFromFolder(data.Dataset):
    def __init__(self, input_folder, Image_list_file = None, subsample=None):
        super(DatasetFromFolder, self).__init__()

        if Image_list_file == None:
            input_files = glob.glob(input_folder + "/*.h5")
        else:
            my_file =  open(Image_list_file, "r")  
            data = my_file.read()  
            data_into_list = data.split("\n")  
            my_file.close()   
            input_files = [os.path.join(input_folder, "%s" % file_name )   for file_name in data_into_list if len(file_name) > 0] 

        input_files.sort() 

        total_length  = len(input_files)

        if subsample is not None:
            num_subsample          = int(subsample*total_length)   
            self.input_files       = random.choices(input_files, k = num_subsample) 
        else:
            self.input_files = input_files

        self.n_images = len(self.input_files) 

    def __getitem__(self, index):

        try:
            h5file = h5py.File(self.input_files[index], 'r') 
        except:
            pdb.set_trace()
  
        inputs = h5file["X"] 
        target = h5file["Y"]  

        inputs  = np.array(inputs)
        target  = np.array(target)
 
        inputs = inputs.astype('float')/256  
        #Warp Hereinput_folder
        #inputs = warp(inputs, WarpUpSampFactor)
        #inputs = np.moveaxis(inputs, 2, 0)
        inputs = torch.from_numpy(inputs)
        
         
        target = target.astype('float')/256   
        #Warp Here
        #target = warp(target, WarpUpSampFactor)
        
        #target = np.moveaxis(target, 2, 0)
        target = torch.from_numpy(target)
         

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
    
    input_X = input_X.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()
    input_Y = input_Y.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()
    Y_predict = Y_predict.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()

 
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
        fig, axs = plt.subplots(1,6,figsize=(15, 3))  

        plot_patch(axs[0], X1, "X1")   
        plot_patch(axs[1], X2, "X2")
        plot_patch(axs[2], X3, "X3")
        plot_patch(axs[3], X4, "X4")
        plot_patch(axs[4], Y, "Y")   
        axs[5].axis('off')
        if text_description is not None:
            axs[5].text(1400, 0,    textstr_sampled,   horizontalalignment='left',    verticalalignment='top',  family='monospace', color="blue") 
    else: 
        fig, axs = plt.subplots(1,7,figsize=(12, 2.75))  
        plot_patch(axs[0], X1, "X1")   
        plot_patch(axs[1], X2, "X2")
        plot_patch(axs[2], X3, "X3")
        plot_patch(axs[3], X4, "X4")
        plot_patch(axs[4], Y,  "Y GT")    
        plot_patch(axs[5], Y_predict, "Y Predict")   
        if text_description is not None:
            axs[6].text(0,  1.0,   textstr_sampled,  horizontalalignment='left',    verticalalignment='top',  family='monospace', color="blue")           
            axs[6].axis('off')

        fig.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath) 

    plt.close("all")   


if __name__ == "__main__":

    INPUTS_PATH = "./CUEE_preprocessing/h5files_Frame-4-Mins/*.h5" 
    dataset = DatasetFromFolder(INPUTS_PATH)
    dataloader = DataLoader(dataset)
    for data_ in dataloader:
        inputs, target = data_    
        long_inputs = torch.cat([inputs,target],dim=1)

        x = inputs
        y = target  
        plotXY(x.permute(2,3,1,0).squeeze(-1), y.permute(2,3,1,0).squeeze(-1))  


