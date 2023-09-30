import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from LiteFlowNet import Network, batch_estimate
from losses import Gradient_Loss, Flow_Loss, Intensity_Loss
from skynet_Unet_model import SkyNet_UNet
from dataLoader import DatasetFromFolder, plotXY
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os

import cupy

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

INPUTS_PATH = "./SkyNet_Data/xTest_skip_pred5.h5"
TARGET_PATH = "./SkyNet_Data/yTest_skip_pred5.h5"

image_path = "./saveplots/prediction_%d.png"

# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(devCount - 1)

device = torch.device(dev if torch.cuda.is_available() else "cpu")


# SkyNet UNet
model = SkyNet_UNet(args.input_channels, args.output_channels) 

# model_name = 'weights/Iteration0.pt'
# model = torch.load(model_name)

model_name = 'weights/weight_%d.pt' % 39
model.load_state_dict(torch.load(model_name)['state_dict'])
model = model.cuda().eval()

# Optical Flow Network
flow_network = Network()
flow_network.load_state_dict(torch.load(lite_flow_model_path))
flow_network.cuda().eval() 

 

testLoader = DataLoader(DatasetFromFolder(INPUTS_PATH, TARGET_PATH), args.BATCH_SIZE, shuffle=False)
 

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

savepath = "results/infer_%s"  % (os.path.basename(model_name).split('.pt')[0])

  
for input_index, (inputs, target) in enumerate(pbar):


    # Training
    inputs = inputs.float().to(device) # The input data
    target = target.float().to(device) 
 
    
    num_images += inputs.size(0) 
    long_inputs = torch.cat([inputs, target],dim=1)
    
    for i in range(5):

        x = long_inputs[:,3*i:(3*i+12),:,:] 
        y = target[:,3*i:(3*i+3),:,:]  
        # Trains model 
 
        with torch.no_grad(): 
            G_output = model(x)
  
        # For Optical Flow 
        input_last = x[:, 9:,:,:].clone().cuda() #I_t
 
        pred_flow_esti_tensor = torch.cat([input_last, G_output],1) #(Predicted)
        gt_flow_esti_tensor = torch.cat([input_last, y],1) #(Ground Truth)
    

        flow_gt    = batch_estimate(gt_flow_esti_tensor, flow_network)
        flow_pred  = batch_estimate(pred_flow_esti_tensor, flow_network)
 

        g_op_loss  = op_loss(flow_pred, flow_gt)  
        g_int_loss = int_loss(G_output, y) 
        g_gd_loss  = gd_loss(G_output, y)

        x = x.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()
        y = y.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()
        G_output = G_output.permute(2,3,1,0).squeeze(-1).detach().cpu().numpy()

        filepath = os.path.join(savepath, "sample_%d.png" % input_index)
        description = {'id': input_index, 'Optical fl loss': g_op_loss.item(), 'Intensity loss': g_int_loss.item(), 'Gradient loss': g_gd_loss.item()}
        savepath = image_path % i
        plotXY(x, y, G_output, savepath=savepath, text_description=description)   
        pbar.set_postfix(description)

 
 