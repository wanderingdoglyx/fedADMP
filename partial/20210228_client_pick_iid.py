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


part_n=[0.5735, 0.5835, 0.6463,0.6686666666666667, 0.7131666666666667, 0.752 ,0.8813333333333334, 0.9563333333333334, 0.9661666666666666 ,0.9716666666666667]
part_b=[0.651, 0.855, 0.90975, 0.913, 0.9325, 0.934, 0.9495, 0.955, 0.9495, 0.9605]
part_k=[0.750, 0.886, 0.929, 0.9505, 0.9665, 0.980, 0.9915, 0.986, 0.9945, 0.989]
part_s=[0.7945, 0.926, 0.958 ,0.974, 0.989 ,0.994 ,0.9945, 0.9985, 0.9935, 0.993]
part_sf=[0.7645, 0.806, 0.848 ,0.914, 0.949 ,0.994 ,0.9945, 0.9985, 0.9935, 0.993]

ratio=[10/100,20/100,30/100,40/100,50/100,60/100,70/100,80/100,90/100,1]

ratio=np.array(ratio)

new_york_time=[28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725,28.91506637059725]
beijing_time=[10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267,10.636886835098267]
KSR_time=[16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184,16.078380016180184]
S_time=[18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544,18.10932878347544]
sf_time=[22.2302, 22.2302, 22.2302, 22.2302, 22.2302, 22.2302, 22.2302,22.2302,22.2302,22.2302]
#batch_ratio=[0.6239949758357954, 0.6933948559944145, 0.6952038879099767, 0.7306343470981064, 0.7485810856578797, 1.0]
batch_ratio=[0.3847243458243244,
 0.4241104852123116,
 0.59504859598075,
 0.6693307305788535,
 0.7230558158522401,
 0.7493739369285396,
 0.8478850736214226,
 0.9002625487336756,
 0.9609293394102465,
 1.0]

n_converage_epoach=[1101,550,367,275,220,183,157,137,122,110]
b_converage_epoach=[800,400,266,200,160,133,114,100,88,80]
k_converage_epoach=[907,453,302,226,181,151,129,113,100,90]
s_converage_epoach=[943,471,314,235,188,157,134,117,104,94]
sf_converage_epoach=[1002,503,343,255,201,169,144,128,114,104]

nt=np.array(new_york_time)*ratio*(100/30)
bt=np.array(beijing_time)*ratio*(100/30)
kt=np.array(KSR_time)*ratio*(100/30)
st=np.array(S_time)*ratio*(100/30)
sft=np.array(sf_time)*ratio*(100/30)

nte=(nt/1200)*np.array(batch_ratio)
bte=(bt/1200)*np.array(batch_ratio)
kte=(kt/1200)*np.array(batch_ratio)
ste=(st/1200)*np.array(batch_ratio)
sfte=(sft/1200)*np.array(batch_ratio)

cut_ratio=[0.5,0.58,0.63,0.68,0.72,0.77,0.81,0.88,0.91,0.97]

nct=nte*n_converage_epoach*cut_ratio
bct=bte*b_converage_epoach*cut_ratio
kct=kte*k_converage_epoach*cut_ratio
sct=ste*s_converage_epoach*cut_ratio
sfct=ste*sf_converage_epoach*cut_ratio

ne=np.array(n_converage_epoach)*cut_ratio
be=np.array(b_converage_epoach)*cut_ratio
ke=np.array(k_converage_epoach)*cut_ratio
se=np.array(s_converage_epoach)*cut_ratio
sfe=np.array(sf_converage_epoach)*cut_ratio
#n_converage_epoach_ratio=[2302/2400,1101/1200,730/800,550/600,440/480,366/400]
#b_converage_epoach_ratio=[1600/2400,800/1200,533/800,400/600,320/480,266/400]
#k_converage_epoach_ratio=[1814/2400,907/1200,604/800,453/600,362/480,302/400]
#s_converage_epoach_ratio=[1686/2400,843/1200,562/800,421/600,337/480,281/400]

#cut_ratio=[0.5,0.58,0.63,0.77,0.88,1]

new_york_acc=[43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917]
beijing_acc=[42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718]
KSR_acc=[44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634]
s_acc=[52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642]
sf_acc=[72.4119,72.4119,72.4119,72.4119,72.4119,72.4119,72.4119,72.4119,72.4119,72.4119]

nc=np.array(new_york_acc)*np.array(part_n)
bc=np.array(beijing_acc)*np.array(part_b)
kc=np.array(KSR_acc)*np.array(part_k)
sc=np.array(s_acc)*np.array(part_s)
sfc=np.array(sf_acc)*np.array(part_sf)

x1=[10,20,30,40,50,60,70,80,90,100]


plt.subplot(1,4,1)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15  # 条形宽度
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
plt.bar(index_female3, height=sfte, width=bar_width, label='San Fransisco')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Epoch time (seconds)',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(a)',font1)
plt.subplot(1,4,2)

waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15  # 条形宽度
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
plt.bar(index_female3, height=sfct, width=bar_width, label='San Fransisco')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(1.10, 1.18), ncol=5,prop = font3)
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Convergence time (seconds)',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(b)',font1)
#############################s t
plt.subplot(1,4,3)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=nc, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=bc, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=kc, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=sc, width=bar_width, color='r', label='Seattle')
plt.bar(index_female3, height=sfc, width=bar_width, label='San Fransisco')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Accuracy % ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(c)',font1)


plt.subplot(1,4,4)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15 # 条形宽度条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=ne, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se, width=bar_width, color='r', label='Seattle')
plt.bar(index_female3, height=sfe, width=bar_width, label='San Fransisco')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Convergence epoch ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(d)',font1)
plt.show()