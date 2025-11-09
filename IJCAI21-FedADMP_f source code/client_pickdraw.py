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

#batch_time_s=[28.73407232325681/1200,15.964918558864145/600,10.671046812974817/400,8.411165547948704/300,6.894216744711716/240,7.674760597970528/200]
batch_ratio=[0.6239949758357954, 0.6933948559944145, 0.6952038879099767, 0.7306343470981064, 0.7485810856578797, 1.0]


nt=np.array(new_york_time)*ratio
bt=np.array(beijing_time)*ratio
kt=np.array(KSR_time)*ratio
st=np.array(S_time)*ratio

ntt=nt*1101/1200
btt=bt*800/1200
ktt=kt*907/1200
stt=st*843/1200


nte=(nt/1200)*np.array(batch_ratio)
bte=(bt/1200)*np.array(batch_ratio)
kte=(kt/1200)*np.array(batch_ratio)
ste=(st/1200)*np.array(batch_ratio)

#converage_epoach=[1101/1200,800/1200,907/1200,843/1200]

new_york_acc=[43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917]
beijing_acc=[42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718]
KSR_acc=[44.0634,44.0634,44.0634,44.0634,44.0634,44.0634]
s_acc=[52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642]

nc=np.array(new_york_acc)*np.array(part_n)
bc=np.array(beijing_acc)*np.array(part_b)
kc=np.array(KSR_acc)*np.array(part_k)
sc=np.array(s_acc)*np.array(part_s)

x1=[5,10,15,20,25,30]


plt.subplot(1,3,1)
waters = ('5', '10', '15','20','25','30')

bar_width = 0.2 # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.tick_params(labelsize=Label_Size)
plt.bar(index_male, height=nte, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=bte, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=kte, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=ste, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Epoch time (seconds)',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(a)',font1)
plt.subplot(1,3,2)
waters = ('5', '10', '15','20','25','30')

bar_width = 0.2 # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.tick_params(labelsize=Label_Size)
plt.bar(index_male, height=nt, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=bt, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=kt, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=st, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4,prop = font3)
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Convergence time (seconds)',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(b)',font1)
#############################s t
plt.subplot(1,3,3)
waters = ('5', '10', '15','20','25','30')


bar_width = 0.2 # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=nc, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=bc, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=kc, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=sc, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Accuracy % ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(c)',font1)
plt.show()