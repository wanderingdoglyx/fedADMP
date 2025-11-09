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

client_time_b=14.4964*1.114
client_time_new_b=14.4964
client_time_k=15.335225411521504*1.110
client_time_new_k=15.335225411521504
client_time_s=15.89129968838849*1.113
client_time_new_s=15.89129968838849
client_time_n=17.51095890421743*1.117
client_time_new_n=17.51095890421743


client_communication_new_b=163529600
client_communication_b=163529600*1.114
client_communication_new_k=220902400
client_communication_k=220902400*1.110
client_communication_new_s=254902400
client_communication_s=254902040*1.113
client_communication_new_n=354419200
client_communication_n=354419200*1.117


Combine_time_n=23.69959691373874
Combine_time_s=15.850149550444772
Combine_time_k=13.910667995158207
Combine_time_b=12.713260218725372

net=0.1963835545686575/100
bet=0.03878962278366089/100
ket=0.060261266720600615/100
set=0.10703109594491814/100
occupation_rate = 0.11
standard_power = 45
power=occupation_rate*standard_power

ET=np.array([net,bet,ket,set])
ET=np.array([Combine_time_n,Combine_time_b,Combine_time_k,Combine_time_s])

G=(cp/google_pixel_4a)*ET
N=(cp/Nokia_TA_1285)*ET
S=(cp/Samsung_Galaxy_S21)*ET
L=(cp/LG_stylo6)*ET

odd=[client_time_n,client_time_b,client_time_k,client_time_s]
new=[client_time_new_n,client_time_new_b,client_time_new_k,client_time_new_s]

odd_p=[client_time_n,client_time_b,client_time_k,client_time_s]
new_p=[client_time_new_n,client_time_new_b,client_time_new_k,client_time_new_s]

new_com=[client_communication_new_n,client_communication_new_b,client_communication_new_k,client_communication_new_s]
odd_com=[client_communication_n,client_communication_b,client_communication_k,client_communication_s]

x1=[10,20,30,40,50,60,70,80,90,100]


plt.subplot(1,3,1)
waters = ('New York', 'Beijing', 'KSP', 'Seattle')

bar_width = 0.13  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标

plt.bar(index_female, height=np.array(odd_p)*power, width=bar_width, color='y', label='Traditional update procedure')
plt.bar(index_male, height=np.array(new_p)*power, width=bar_width, color='b', label='New update procedure')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5,prop = font3)  # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J',font1)  # 纵坐标轴标题
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(a)',font1)

plt.subplot(1,3,2)

waters = ('New York', 'Beijing', 'KSP', 'Seattle')

bar_width = 0.13  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标

plt.bar(index_female, height=odd, width=bar_width, color='y', label='Traditional update procedure')
plt.bar(index_male, height=new, width=bar_width, color='b', label='New update procedure')

plt.tick_params(labelsize=Label_Size)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5,prop = font3)  # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Time (seconds)',font1)  # 纵坐标轴标题
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(b)',font1)
#############################s t
plt.subplot(1,3,3)
waters = ('New York', 'Beijing', 'KSP', 'Seattle')

bar_width = 0.13  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标

plt.bar(index_female, height=np.array(odd_com)/10e6, width=bar_width, color='y', label='Traditional update procedure')
plt.bar(index_male, height=np.array(new_com)/10e6, width=bar_width, color='b', label='New update procedure')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5,prop = font3)  # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB',font1)  # 纵坐标轴标题
#plt.title('购买饮用水情况的调查结果')  # 图形标题

plt.title('(c)',font1)

plt.show()


