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

part_n=[0.8996666666666667, 0.9646666666666667, 0.9758333333333333, 0.9878333333333333, 0.9986666666666666, 1]
part_b=[0.9515, 0.987, 0.9995, 0.997, 0.999, 1]
part_k=[0.901, 0.9385, 0.9515, 0.986, 0.9925, 1]
part_s=[0.9555, 0.968, 0.989, 0.9915, 0.9925, 1]
ratio=[5/30,10/30,15/30,20/30,25/30,1]

part_n_noniid=[0.5435, 0.5471666666666667, 0.5731666666666667, 0.7096666666666667, 0.9045, 0.9716666666666667]
part_b_noniid=[0.657, 0.805, 0.8325, 0.8505 ,0.9595, 0.9605]
part_k_noniid=[0.742, 0.8325 ,0.8665, 0.8915, 0.9885, 0.989]
part_s_noniid=[0.7555, 0.838, 0.889, 0.9315, 0.9925, 0.993]

ratio=np.array(ratio)

new_york_time=[58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725]
beijing_time=[11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267]
KSR_time=[18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184]
S_time=[32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544]

batch_ratio=[0.6239949758357954, 0.6933948559944145, 0.6952038879099767, 0.7306343470981064, 0.7485810856578797, 1.0]

#n_converage_epoach=[1802*(),1101*((10*10)/(10*60)),730,550,430,360]
#b_converage_epoach=[1600,800,533,400,320,266]
#k_converage_epoach=[1814,907,604,453,362,302]
#s_converage_epoach=[1886,943,662,521,437,381]

n_converage_epoach=[200,183,169,157,146,137]
b_converage_epoach=[145,133,123,114,106,100]
k_converage_epoach=[164,151,139,129,120,113]
s_converage_epoach=[171,157,145,134,125,117]

nt=np.array(new_york_time)*ratio
bt=np.array(beijing_time)*ratio
kt=np.array(KSR_time)*ratio
st=np.array(S_time)*ratio

nte=(nt/1200)*np.array(batch_ratio)+np.array([0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761])
bte=(bt/1200)*np.array(batch_ratio)+np.array([0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138])
kte=(kt/1200)*np.array(batch_ratio)+np.array([0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207])
ste=(st/1200)*np.array(batch_ratio)+np.array([0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457])

cut_ratio=[0.76,0.78,0.83,0.87,0.88,0.91]
#cut_ratio=[1,1,1,1,1,1]

nct=nte*n_converage_epoach*cut_ratio
bct=bte*b_converage_epoach*cut_ratio
kct=kte*k_converage_epoach*cut_ratio
sct=ste*s_converage_epoach*cut_ratio

ne=np.array(n_converage_epoach)*cut_ratio
be=np.array(b_converage_epoach)*cut_ratio
ke=np.array(k_converage_epoach)*cut_ratio
se=np.array(s_converage_epoach)*cut_ratio


#n_converage_epoach_ratio=[2302/2400,1101/1200,730/800,550/600,440/480,366/400]
#b_converage_epoach_ratio=[1600/2400,800/1200,533/800,400/600,320/480,266/400]
#k_converage_epoach_ratio=[1814/2400,907/1200,604/800,453/600,362/480,302/400]
#s_converage_epoach_ratio=[1686/2400,843/1200,562/800,421/600,337/480,281/400]

cut_ratio=[0.5,0.58,0.63,0.77,0.88,1]

new_york_acc=[43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917]
beijing_acc=[42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718]
KSR_acc=[44.0634,44.0634,44.0634,44.0634,44.0634,44.0634]
s_acc=[52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642]

nc=np.array(new_york_acc)*np.array(part_n)
bc=np.array(beijing_acc)*np.array(part_b)
kc=np.array(KSR_acc)*np.array(part_k)
sc=np.array(s_acc)*np.array(part_s)


x1=[5,10,15,20,25,30]


plt.subplot(1,4,1)
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
plt.ylabel('Average epoch time (seconds)',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(a)',font1)
plt.subplot(1,4,2)

waters = ('5', '10', '15','20','25','30')

bar_width = 0.2 # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.tick_params(labelsize=Label_Size)
plt.bar(index_male, height=nct, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=bct, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=kct, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=sct, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(1.10, 1.18), ncol=4,prop = font3)
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Convergence time (seconds)',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(b)',font1)
#############################s t
plt.subplot(1,4,3)
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


plt.subplot(1,4,4)
waters = ('5', '10', '15','20','25','30')


bar_width = 0.2 # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=ne, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Convergence epoch ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(d)',font1)
plt.show()