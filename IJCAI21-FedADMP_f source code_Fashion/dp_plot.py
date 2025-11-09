import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from datetime import datetime

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

x=[10,20,40,60,80,100]
nyc_acc=[ 8.5737125, 14.57485833, 23.09908333, 25.36730139, 29.378475,   31.68725]#,46.10810417]
beijing_acc= [2.94296169, 11.60742897, 25.34957724,  31.69256901,
 38.73847747, 42.05401232]#, 43.1225235]
k_acc=[2.72884551, 13.68286563, 28.91474684, 38.75032229, 42.31306332, 44.90201859]#,47.776608 ]
seattle_acc=[ 3.5360785,  15.84069689, 33.03493125, 44.34096778, 49.44864181, 52.05252]#,59.1822 ]

plt.xlabel('differential privacy $\epsilon$ ',font1)  # x轴变量名称
plt.ylabel('Convergence accuracy %',font1)
plt.tick_params(labelsize=Label_Size)

plt.plot(x,nyc_acc, Colors[0], label=Labels[0], marker='s',linewidth=3)
plt.plot(x,beijing_acc,Colors[1], label=Labels[1], marker='8',linewidth=3)
plt.plot(x,k_acc,Colors[2], label=Labels[2],marker='p', linewidth=3)
plt.plot(x,seattle_acc, Colors[3], label=Labels[3],marker='P', linewidth=3)


plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5,prop = font3) # 显示图例

plt.grid( linestyle='--')

plt.show()



