import matplotlib
from matplotlib import pyplot as plt
import sys
import numpy as np
import statsmodels.api as sm
########################################################

###########################################################
Label_Size = 30  # 刻度的大小
Title_Size = 40
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
font1 = {'weight': 'normal', 'size': 30}
font2 = {'weight': 'normal', 'size': 30}
font3 = {'weight': 'normal', 'size': 40}


Names = ['you_F', 'osdi_F', 'caca_F', 'iqiyi_F']
Labels = ['New York', 'Beijing', 'Krong Siem Reap', 'Seattle']
Colors = ['y', 'b', 'g', 'r','c']




cb1=[ 4.42341675,20.843325,28.26010955,44.01472897,44.79217805,44.9105633 ]
ck1=[ 9.82564853,25.87,32.49790866,46.07896094,46.90751108,46.90751108]
cs1=[ 5.3536345, 30.55,47.932092 , 56.828942 ,59.1452645,59.5452645]
cn1=[ 4.388225 ,31.54,39.2886 ,  46.5811,   48.479725,48.479725]
x1=[32,50,64,100,128,256]


fig, (ax1) = plt.subplots(1, 1, figsize=(22, 10))
############data_time_interveral

    ############Beijing

Ln1 = ax1.plot(x1,cn1 , Colors[0], label=Labels[0], marker='s',linewidth=3)
Ln2 = ax1.plot(x1,cb1 , Colors[1], label=Labels[1], marker='8',linewidth=3)
Ln3 = ax1.plot(x1,ck1 , Colors[2], label=Labels[2],marker='p', linewidth=3)
Ln4 = ax1.plot(x1,cs1 , Colors[3], label=Labels[3],marker='P', linewidth=3)

ax1.set_xlabel("Hidden size", font1)
ax1.set_ylabel("Convergence accuracy %", font1)
ax1.tick_params(labelsize=Label_Size)
ax1.grid(linestyle='--')
#ax1.set_title("(a)", y=-0.3, fontsize=Title_Size)
fig.legend([Ln1, Ln2, Ln3, Ln4],  # The line objects
           labels=Labels,  # The labels for each line
           loc="upper center",  # Position of legend
           ncol=6,
           borderaxespad=0.1,  # Small spacing around legend box
           prop=font3,
           )

plt.show()


