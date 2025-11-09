import matplotlib
from matplotlib import pyplot as plt
import sys
import numpy as np
import statsmodels.api as sm

Label_Size = 20  # 刻度的大小
Title_Size = 35
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
font1 = {'weight': 'normal', 'size': 25}
font2 = {'weight': 'normal', 'size': 25}
font3 = {'weight': 'normal', 'size': 35}


Names = ['you_F', 'osdi_F', 'caca_F', 'iqiyi_F']
Labels = ['New York', 'Beijing', 'Krong Siem Reap', 'Seattle']
Colors = ['y', 'b', 'g', 'r','c']

google_pixel_4a=2.2
Nokia_TA_1285=1.8
Samsung_Galaxy_S21=2.84
LG_stylo6=2.3
cp=2.6

Combine_time_n=23.69959691373874
Combine_time_s=15.850149550444772
Combine_time_k=13.910667995158207
Combine_time_b=12.713260218725372

net=0.1963835545686575/100
bet=0.03878962278366089/100
ket=0.060261266720600615/100
set=0.10703109594491814/100

ET=np.array([net,bet,ket,set])
ET=np.array([Combine_time_n,Combine_time_b,Combine_time_k,Combine_time_s])

G=(cp/google_pixel_4a)*ET
N=(cp/Nokia_TA_1285)*ET
S=(cp/Samsung_Galaxy_S21)*ET
L=(cp/LG_stylo6)*ET

#####################Energy comsumption when reaching 20% accuracy
waters = ('New York', 'Beijing', 'Krong Siem Reap', 'Seattle')

bar_width = 0.13  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
#index_female4 = index_female3 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.bar(index_female2, height=G, width=bar_width, color='y', label='Google pixel 4a')
plt.bar(index_male, height=N, width=bar_width, color='b', label='Nokia TA1285')
plt.bar(index_female, height=S, width=bar_width, color='g', label='Samsung Galaxy S21')
plt.bar(index_female1, height=L, width=bar_width, color='r', label='LG stylo6')
#plt.bar(index_female3, height=Combine_energy, width=bar_width,color='c', label='FedADMP')
plt.tick_params(labelsize=Label_Size)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5,prop = font3)  # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Time (seconds)',font1)  # 纵坐标轴标题
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.show()



