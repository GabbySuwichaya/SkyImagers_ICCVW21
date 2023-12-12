import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from LiteFlowNet import Network, batch_estimate
from losses import Gradient_Loss, Flow_Loss, Intensity_Loss
from skynet_Unet_model import SkyNet_UNet
from dataloader_CUEE import DatasetFromFolder, plotXY, PSNR
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import pdb 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--input_channels',
                    default=12, 
                    help='(default value: %(default)s)https://github.com/GabbySuwichaya/SkyImagers_ICCVW21/tree/main/weights Number of channels for input images. input_channels = 3*NumOfImages')
parser.add_argument('--output_channels',
                    default=3, 
                    help='(default value: %(default)s) Number of channels for output images.')
parser.add_argument('--lam_int',
                    default=5.0, 
                    help='(default value: %(default)s) Hyperparameter for intensity loss.')
parser.add_argument('--lam_gd',
                    default=0.00111, 
                    help='(default value: %(default)s) Hyperparameter for gradient loss.')
parser.add_argument('--lam_op',
                    default=0.010, 
                    help='(default value: %(default)s) Hyperparameter for optical flow loss.')
parser.add_argument('--EPOCHS',
                    default=40, 
                    help='(default value: %(default)s) Number of epochs o train model for.')
parser.add_argument('--BATCH_SIZE',
                    default=1, 
                    help='(default value: %(default)s) Training batch size.')
parser.add_argument('--LR',
                    default=0.0002, 
                    help='(default value: %(default)s) learning rate.')

args = parser.parse_args()


#Model Paths
lite_flow_model_path='./network-sintel.pytorch'

#INPUTS_PATH = "CUEE_preprocessing/Testing-Tr0p80-Val0p00-Test0p20/h5files_Frame-4-Mins_IMS-1920x1080"
INPUTS_PATH = "CUEE_preprocessing/Saranphat/h5files_Frame-4-Mins_IMS-1920x1080-Saranphat"


# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(devCount - 1)

device = torch.device(dev if torch.cuda.is_available() else "cpu")


# SkyNet UNet
model = SkyNet_UNet(args.input_channels, args.output_channels) 

new_weight = True 

if new_weight == True: 
    ep = 39
    model_name = 'train_weights_Tr0p80-Val0p00-Test0p20_480_Shuffle/weight_%03d.pt' % ep
    prediction_folder = "saveimage_CUEE_weights_%03d" %  ep
else:
    ep = 20
    model_name = 'weights/backups/weight_ %d.pt' % ep
    prediction_folder = "saveimage_TSI_weights_%02d" % ep

os.makedirs( prediction_folder, exist_ok=True) 


#model_name = 'weights/weight_39.pt' 
model.load_state_dict(torch.load(model_name)['state_dict'])
model = model.cuda().eval()

# Optical Flow Network
flow_network = Network()
flow_network.load_state_dict(torch.load(lite_flow_model_path))
flow_network.cuda().eval() 

  
testLoader = DataLoader(DatasetFromFolder(INPUTS_PATH ), args.BATCH_SIZE, shuffle=False)
 

# Training Loss
train_loss = []

# Validation Loss 
valid_loss = []

# Losses
gd_loss  = Gradient_Loss(1, 3).to(device)
op_loss  = Flow_Loss().to(device)
int_loss = Intensity_Loss(1).to(device)


trainLossCount = 0
num_images = 0
pbar = tqdm(testLoader)

os.makedirs( "results_cuee_trained_480", exist_ok=True) 
image_path = "results_cuee_trained_480/prediction_%d.png"


psnr_all = []  
for input_index, (inputs, target, image_name) in enumerate(pbar):


    # Training
    inputs = inputs.float().to(device) # The input data
    target = target.float().to(device)  
         
    x = inputs
    y_groundtruth = target 

    with torch.no_grad(): 
        y_predict = model(x) 
 
    Image_gt   = y_groundtruth.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy() 
    Image_pred = y_predict.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()

    Image_gt   = (Image_gt - Image_gt.min())/(Image_gt.max() - Image_gt.min())
    Image_pred = (Image_pred - Image_pred.min())/(Image_pred.max() - Image_pred.min())

    plt.imsave(os.path.join(prediction_folder, image_name[0] + "-gt.png"), Image_gt)
    
    plt.imsave(os.path.join(prediction_folder, image_name[0] + "-pred.png"), Image_pred)
    
    psnr = PSNR(Image_gt*255, Image_pred*255)
    

    txtf = open(os.path.join(prediction_folder, image_name[0] + '-PSNR.txt'),'w+')
    txtf.write("%s:%f" % (image_name[0], psnr)  ) 
    txtf.close()

    psnr_all.append(psnr)

    description = {'id': input_index, "psnr":psnr}
    savepath = image_path % input_index 
         
    pbar.set_postfix(description) 


print("Average-PSNR %f" % (sum(psnr_all)/len(psnr_all)))

 
 