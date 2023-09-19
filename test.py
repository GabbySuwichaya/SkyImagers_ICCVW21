import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from LiteFlowNet import *
from losses import *
from skynet_Unet_model import *
from dataLoader import *
from torch.utils.data import DataLoader
import argparse
import numpy as np

import cupy

parser = argparse.ArgumentParser()

parser.add_argument('--input_channels',
                    default=12, 
                    help='(default value: %(default)s) Number of channels for input images. 3*NumOfImages')
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

INPUTS_PATH = "./SkyNet_Data/xTrain_skip.h5"
TARGET_PATH = "./SkyNet_Data/yTrain_skip.h5"


# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(devCount - 1)

device = torch.device(dev if torch.cuda.is_available() else "cpu")


# SkyNet UNet
# model = SkyNet_UNet(args.input_channels, args.output_channels) 
model = torch.load('weights/Iteration0.pt')
model = model.eval()

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
gd_loss = Gradient_Loss(1, 3).to(device)
op_loss = Flow_Loss().to(device)
int_loss = Intensity_Loss(1).to(device)


trainLossCount = 0
num_images = 0
pbar = tqdm(testLoader)

pdb.set_trace()

for i, data in enumerate(pbar):
    # Training
    inputs = Variable(data[0]).to(device) # The input data
    target = Variable(data[1]).float().to(device)
    
    num_images += inputs.size(0)
    
    # Trains model 
    with torch.no_grad(): 
        G_output = model(inputs)
  
    # For Optical Flow 
    input_last = inputs[:, 9:,:,:].clone().cuda() #I_t
 
    pred_flow_esti_tensor = torch.cat([input_last, G_output],1) #(Predicted)
    gt_flow_esti_tensor = torch.cat([input_last, target],1) #(Ground Truth)
    

    flow_gt    = batch_estimate(gt_flow_esti_tensor, flow_network)
    flow_pred  = batch_estimate(pred_flow_esti_tensor, flow_network)
 
    g_op_loss  = op_loss(flow_pred, flow_gt)
    g_int_loss = int_loss(G_output, target)
    g_gd_loss  = gd_loss(G_output, target)
 
    pbar.set_postfix({'Optial flow loss': g_op_loss.item(), 'Intensity loss': g_int_loss.item(), 'Gradient loss': g_gd_loss.item()})

 
 