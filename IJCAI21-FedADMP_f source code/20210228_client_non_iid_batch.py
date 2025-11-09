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


part_n=[0.5435, 0.5435, 0.5463,0.5486666666666667, 0.5731666666666667, 0.622 ,0.7513333333333334, 0.8563333333333334, 0.9461666666666666 ,0.9716666666666667]
part_b=[0.631, 0.845, 0.8975, 0.893, 0.9325, 0.934, 0.9495, 0.955, 0.9495, 0.9605]
part_k=[0.730, 0.856, 0.919, 0.9505, 0.9665, 0.980, 0.9915, 0.986, 0.9945, 0.989]
part_s=[0.7645, 0.886, 0.948 ,0.974, 0.989 ,0.994 ,0.9945, 0.9985, 0.9995, 0.9999]
ratio=[10/100,20/100,30/100,40/100,50/100,60/100,70/100,80/100,90/100,1]

part_n_noniid=[0.6991666666666667, 0.8313333333333334, 0.895, 0.9243333333333333, 0.9336666666666666, 0.9413333333333335, 0.9451666666666667, 0.9505, 0.9578333333333333, 0.9540000000000001]
part_b_noniid=[0.8300000000000001, 0.858, 0.89, 0.934, 0.9640000000000001, 0.9784999999999999, 0.99, 0.988, 0.9855, 0.993]
part_k_noniid=[0.68, 0.765, 0.8475, 0.8840000000000001, 0.9055, 0.9159999999999999, 0.9195000000000001, 0.9085, 0.922, 0.941]
part_s_noniid=[0.7595000000000001, 0.8115000000000001, 0.8645, 0.9205, 0.9505, 0.941, 0.9720000000000001, 0.98, 0.991, 0.996]
ratio=np.array(ratio)

new_york_time=[58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725,58.91506637059725]
beijing_time=[11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267,11.636886835098267]
KSR_time=[18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184,18.078380016180184]
S_time=[32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544,32.10932878347544]

batch_ratio=[0.3847243458243244,
 0.5041104852123116,
 0.59504859598075,
 0.6693307305788535,
 0.7230558158522401,
 0.7493739369285396,
 0.8478850736214226,
 0.9002625487336756,
 0.9609293394102465,
 1.0]
#n_converage_epoach=[1802,1101,730,550,430,360]
#b_converage_epoach=[1600,800,533,400,320,266]
#k_converage_epoach=[1814,907,604,453,362,302]
#s_converage_epoach=[1886,943,662,521,437,381]
#n_converage_epoach=[198,171,148,132,124,113,98,84,75,72]

n_converage_epoach=[1190,600,397,299,249,208,168,137,118,108]
b_converage_epoach=[810,410,276,210,170,130,110,97,87,74]
k_converage_epoach=[917,463,312,236,181,150,125,110,95,85]
s_converage_epoach=[953,481,324,245,188,157,130,110,97,84]
batch_epoch=[10/60,20/70,30/80,40/90,50/100,60/110,70/120,80/130,90/140,100/150]


nt=np.array(new_york_time)*ratio*(100/23)
bt=np.array(beijing_time)*ratio*(100/23)
kt=np.array(KSR_time)*ratio*(100/23)
st=np.array(S_time)*ratio*(100/23)

nte=(nt/1200)*np.array(batch_ratio)+np.array([0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761])
bte=(bt/1200)*np.array(batch_ratio)+np.array([0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138])
kte=(kt/1200)*np.array(batch_ratio)+np.array([0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207])
ste=(st/1200)*np.array(batch_ratio)+np.array([0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457])

#cut_ratio=[0.5,0.58,0.63,0.77,0.88,0.91]
cut_ratio=[1,1,1,1,1,1,1,1,1,1]

nct=nte*n_converage_epoach*cut_ratio*batch_epoch
bct=bte*b_converage_epoach*cut_ratio*batch_epoch
kct=kte*k_converage_epoach*cut_ratio*batch_epoch
sct=ste*s_converage_epoach*cut_ratio*batch_epoch

print(list(nct),'nct')
print(list(bct),'bct')
print(list(kct),'kct')
print(list(sct),'sct')

ne=np.array(n_converage_epoach)*cut_ratio*batch_epoch
be=np.array(b_converage_epoach)*cut_ratio*batch_epoch
ke=np.array(k_converage_epoach)*cut_ratio*batch_epoch
se=np.array(s_converage_epoach)*cut_ratio*batch_epoch
print(list(ne),'ne')
print(list(be),'be')
print(list(ke),'ke')
print(list(se),'se')


new_york_acc=[43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917]
beijing_acc=[42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718]
KSR_acc=[44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634]
s_acc=[52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642]

nc=np.array(new_york_acc)*np.array(part_n)*part_n_noniid
bc=np.array(beijing_acc)*np.array(part_b)*part_b_noniid
kc=np.array(KSR_acc)*np.array(part_k)*part_k_noniid
sc=np.array(s_acc)*np.array(part_s)*part_s_noniid


x1=[10,20,30,40,50,60,70,80,90,100]


plt.subplot(1,4,1)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
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

waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
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
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
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
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
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