import glob
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pdb

import torch
from skynet_Unet_model import SkyNet_UNet
from dataloader_CUEE import DatasetFromFolder, plotXY, PSNR
from torch.utils.data import DataLoader
import argparse
from LiteFlowNet import Network, batch_estimate
from losses import Gradient_Loss, Flow_Loss, Intensity_Loss


from tqdm import tqdm
# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(devCount - 1)

device = torch.device(dev if torch.cuda.is_available() else "cpu")



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

parser.add_argument('--image_size',
                    default=480, 
                    help='480 or 512')

parser.add_argument('--data_setting',
                    default=[80,0,20], 
                    help='settings of the testing')

args = parser.parse_args()

  
  

Data_Setting     = "Tr0p%02d-Val0p%02d-Test0p%02d" % (args.data_setting[0], args.data_setting[1], args.data_setting[2] )
INPUTS_PATH      = "./CUEE_preprocessing/h5files_Frame-4-Mins_IMS-%d" %    args.image_size
Image_list_file  = "./CUEE_preprocessing/test_data_%s.txt" % Data_Setting
 

model = SkyNet_UNet(args.input_channels, args.output_channels)
 

# Optical Flow Network
lite_flow_model_path='./network-sintel.pytorch'
flow_network = Network()
flow_network.load_state_dict(torch.load(lite_flow_model_path))
flow_network.cuda().eval() 


# Losses
gd_loss = Gradient_Loss(1, 3).to(device)
op_loss = Flow_Loss().to(device)
int_loss = Intensity_Loss(1).to(device)

 


epoch_loss_valid = []
epoch_PSNR_valid = [] 

epoch_loss_train = []
epoch_PSNR_train = []

testLoader = DataLoader(DatasetFromFolder(INPUTS_PATH, Image_list_file=Image_list_file, subsample=0.2),  args.BATCH_SIZE, shuffle=True) 


for  ep_ in range(13):

    model_name       = 'train_weights_%s_%d_Shuffle/weight_%03d.pt' %  (Data_Setting, args.image_size, ep_)

    torch_loaded = torch.load(model_name)

    model.load_state_dict(torch_loaded['state_dict'])
    model = model.cuda().eval()  
     
    epoch_loss_train.append(torch_loaded['train_loss'])
    epoch_PSNR_train.append(torch_loaded['train_psnr'])
    
    valid_PSNR = []
    valid_loss = [] 

    pbar = tqdm(testLoader)

    for input_index, (inputs, target) in enumerate(pbar):


        # Training
        inputs = inputs.float().to(device) # The input data
        target = target.float().to(device)  
            
        x = inputs
        y = target 

        with torch.no_grad(): 
            G_output = model(x) 
    
        # For Optical Flow
        inputs     = x 
        input_last = inputs[:, 9:,:,:].clone().cuda() #I_t 

        pred_flow_esti_tensor = torch.cat([input_last, G_output],1) #(Predicted)
        gt_flow_esti_tensor = torch.cat([input_last, target],1) #(Ground Truth) 

        flow_gt   = batch_estimate(gt_flow_esti_tensor, flow_network)
        flow_pred = batch_estimate(pred_flow_esti_tensor, flow_network)
        
        g_op_loss = op_loss(flow_pred, flow_gt)
        g_int_loss = int_loss(G_output, target)
        g_gd_loss = gd_loss(G_output, target)
    
        g_loss = args.lam_gd*g_gd_loss + args.lam_op*g_op_loss + args.lam_int*g_int_loss

        valid_loss.append(g_loss.item())
        
        PSNR_BxC_temp = PSNR(G_output.cpu().numpy(), target.cpu().numpy())   
        valid_PSNR.append(PSNR_BxC_temp)
        pbar.set_postfix({'Valid loss': g_loss.item(), "PSNR" :PSNR_BxC_temp})
 

    epoch_loss_valid.append(sum(valid_loss)/len(valid_loss) )
    epoch_PSNR_valid.append(sum(valid_PSNR)/len(valid_PSNR) )

########################################################################## 
 
plt.figure()
plt.plot(epoch_PSNR_train, label="PSNR [training]",   color='blue', linestyle = "-", linewidth=2)
plt.plot(epoch_PSNR_valid, label="PSNR [validation]", color='red',  linestyle = "--", linewidth=2) 
plt.xlabel('Iterations')
plt.ylabel('Loss function value') 
#plt.xlim([0,500])
plt.legend()
plt.grid()                                     # draw grid for major ticks
plt.grid(which='minor', alpha=0.3)   
plt.savefig("Valid_vs_train.png")
plt.show()

pdb.set_trace()



