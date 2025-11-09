import matplotlib
from matplotlib import pyplot as plt
import sys
import numpy as np
import statsmodels.api as sm

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

part_n=[0.8596666666666667, 0.9346666666666667, 0.9458333333333333, 0.9578333333333333, 0.9586666666666666, 1]
part_b=[0.9015, 0.982, 0.9895, 0.996, 0.991, 1]
part_k=[0.771, 0.8985, 0.9215, 0.946, 0.9525, 1]
part_s=[0.8555, 0.938, 0.989, 0.9915, 0.9925, 1]
ratio=[5/30,10/30,15/30,20/30,25/30,1]

ratio=np.array(ratio)

new_york_time=[58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725]
beijing_time=[11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267]
KSR_time=[18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184]
S_time=[32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544]

batch_ratio=[0.6239949758357954, 0.6933948559944145, 0.6952038879099767, 0.7306343470981064, 0.7485810856578797, 1.0]


nt=np.array(new_york_time)*ratio
bt=np.array(beijing_time)*ratio
kt=np.array(KSR_time)*ratio
st=np.array(S_time)*ratio

nte=(nt/1200)*np.array(batch_ratio)
bte=(bt/1200)*np.array(batch_ratio)
kte=(kt/1200)*np.array(batch_ratio)
ste=(st/1200)*np.array(batch_ratio)

#n_converage_epoach_ratio=[2302/2400,1101/1200,730/800,550/600,440/480,366/400]
#b_converage_epoach_ratio=[1600/2400,800/1200,533/800,400/600,320/480,266/400]
#k_converage_epoach_ratio=[1814/2400,907/1200,604/800,453/600,362/480,302/400]
#s_converage_epoach_ratio=[1686/2400,843/1200,562/800,421/600,337/480,281/400]

n_converage_epoach=[2302,1101,730,550,440,366]
b_converage_epoach=[1600,800,533,400,320,266]
k_converage_epoach=[1814,907,604,453,362,302]
s_converage_epoach=[1686,843,562,421,337,281]
