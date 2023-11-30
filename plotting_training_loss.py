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
 

lam_int = 5.00
lam_gd  = 0.00111
lam_op  = 0.010

total_loss  = df["total_loss"].values
op_loss     = lam_op*df["op_loss"].values
int_loss    = lam_int*df["int_loss"].values
gd_loss     = lam_op*df["gd_loss"].values

plt.figure()
plt.plot(total_loss, label="Total loss [training]", color='black', linestyle = "-", linewidth=2)
plt.plot(op_loss, label="Optical loss [training]",  color='orange', linestyle = "-.",linewidth=2, alpha=0.5)
plt.plot(int_loss, label="Intensity loss [training]",    color='lime', linestyle = "--",linewidth=2, alpha=0.5)
plt.plot(gd_loss, label="Gradient loss [training]",      color='blue', linestyle = ":",linewidth=2, alpha=0.5)
plt.xlabel('Iterations')
plt.ylabel('Loss function value') 
#plt.xlim([0,500])
plt.legend()
plt.grid()                                     # draw grid for major ticks
plt.grid(which='minor', alpha=0.3)   
plt.show()

pdb.set_trace()



