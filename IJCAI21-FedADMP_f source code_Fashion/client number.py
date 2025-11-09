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


x=[1,3,9,15,30]
clients_b=[44.52897832 ,44.47911517 ,42.7679647 , 42.42971995 ,42.30507718]
clients_time_b=[190.1407835483551,95.08784461021423,39.82355737686157,22.967611074447632, 11.636886835098267]

clients_k=[48.29758115,48.39307706,45.2540755, 44.52564209,44.0634]
clients_time_k=[267.4263651187603,132.67023998040418,59.4532563319573,33.20655113000136, 18.078380016180184]

clients_s=[57.3050775,  57.655856,   55.800874 ,  53.55680758,53.48]
clients_time_s=[333.06607554509094,170.46314885066107,72.9662031026987,42.847320996798004,32.10932878347544]

clients_n=[47.33841667, 46.82071667 ,44.32728333, 43.417,     43.38262917]
clients_time_n=[452.5309283549969,224.42739233603842,93.64575950915996,88.16749541576092,58.91506637059725]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
############data_time_interveral

    ############Beijing

Ln1 = ax1.plot(x,clients_n , Colors[0], label=Labels[0], marker='s',linewidth=3)
Ln2 = ax1.plot(x,clients_b , Colors[1], label=Labels[1], marker='8',linewidth=3)
Ln3 = ax1.plot(x,clients_k , Colors[2], label=Labels[2],marker='p', linewidth=3)
Ln4 = ax1.plot(x,clients_s , Colors[3], label=Labels[3],marker='P', linewidth=3)

ax1.set_xlabel("Client number ", font1)
ax1.set_ylabel("Accuracy %", font1)
ax1.tick_params(labelsize=Label_Size)
ax1.grid(linestyle='--')
ax1.set_title("(a)", y=-0.3, fontsize=Title_Size)

###################krong
ax2.plot(x,clients_time_n , Colors[0], label=Labels[0],marker='s', linewidth=3)
ax2.plot(x,clients_time_b , Colors[1], label=Labels[1],marker='8', linewidth=3)
ax2.plot(x, clients_time_k, Colors[2], label=Labels[2],marker='p', linewidth=3)
ax2.plot(x,clients_time_s , Colors[3], label=Labels[3],marker='P', linewidth=3)

ax2.set_xlabel("Client number ", font1)
ax2.set_ylabel("Time (Seconds)", font1)
ax2.tick_params(labelsize=Label_Size)
ax2.grid(linestyle='--')
ax2.set_title("(b)", y=-0.3, fontsize=Title_Size)
fig.legend([Ln1, Ln2, Ln3, Ln4],  # The line objects
           labels=Labels,  # The labels for each line
           loc="upper center",  # Position of legend
           ncol=6,
           borderaxespad=0.1,  # Small spacing around legend box
           prop=font3,
           )

fig.tight_layout()  # 调整整体空白

# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
plt.subplots_adjust(top=0.85, wspace=0.25, hspace=0.4)

# save_path = "/Users/gangyan/Desktop/RLB_Data/Figures/data_describe.pdf"
# plt.savefig(save_path, dpi=1400)

plt.show()