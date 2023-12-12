import glob
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pdb

Root_file      = "train_loss_Tr0p80-Val0p00-Test0p20_480_Shuffle" 
pickle_files =   ["trainloss_epoch_%03d_15000.pickle" % ep for ep in [0, 1, 2, 3, 4, 5, 6   ]] 

total_loss_list = []
for pickle_ in pickle_files:

    file_path = os.path.join(Root_file, pickle_)

    if os.path.isfile(file_path):
        file_to_read = open(file_path, "rb")

        loaded_dictionary = pickle.load(file_to_read)
        total_loss_list  += loaded_dictionary
 

df = pd.DataFrame(total_loss_list) 
 

lam_int = 0.5
lam_gd  = 0.00111
lam_op  = 0.010

total_loss  = df["total_loss"].values
op_loss     = lam_op*df["op_loss"].values
int_loss    = lam_int*df["int_loss"].values
gd_loss     = lam_op*df["gd_loss"].values

fig, ax = plt.subplots(4, 1,  sharex=True,  figsize=(7, 5))
ax[0].plot(total_loss, label="Total loss", color='black', linestyle = "-", linewidth=2)
ax[0].set_ylabel('Total Loss')
ax[0].grid()      

ax[1].plot(int_loss, label="Intensity loss [training]",    color='green', linestyle = "--",linewidth=2, alpha=0.5)
ax[1].set_ylabel('Intensity Loss')
ax[1].grid()      


ax[2].plot(op_loss, label="Optical loss [training]",  color='orange', linestyle = "-.",linewidth=2, alpha=0.5)
ax[2].set_ylabel('Optical Loss')
ax[2].grid()      


ax[3].plot(gd_loss, label="Gradient loss [training]",      color='blue', linestyle = ":",linewidth=2, alpha=0.5)
ax[3].set_ylabel('Gradient Loss')
ax[3].set_xlabel('Iterations')
ax[3].grid()      
ax[3].set_xlim([0,500])
# draw grid for major ticks  
plt.savefig("Loss")

pdb.set_trace()



