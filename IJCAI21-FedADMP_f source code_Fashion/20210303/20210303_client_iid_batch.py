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
part_b=[0.601, 0.805, 0.8675, 0.893, 0.9025, 0.924, 0.9395, 0.955, 0.9595, 0.9605]
part_k=[0.700, 0.816, 0.909, 0.9305, 0.9565, 0.980, 0.9915, 0.986, 0.9945, 0.989]
part_s=[0.7445, 0.856, 0.918 ,0.944, 0.969 ,0.994 ,0.9945, 0.9985, 0.9995, 0.9999]
ratio=[10/100,20/100,30/100,40/100,50/100,60/100,70/100,80/100,90/100,1]

part_n_noniid=[0.44133333333333336, 0.7334999999999999, 0.8161666666666666, 0.913, 0.9048333333333334, 0.9308333333333333, 0.9443333333333334, 0.9463333333333335, 0.9455, 0.9458333333333333]
part_b_noniid=[0.7384999999999999, 0.8375, 0.8945000000000001, 0.898, 0.9485, 0.9720000000000001, 0.975, 0.99, 0.9875, 0.9895]
part_k_noniid=[0.4515, 0.6875, 0.7825, 0.8565, 0.8740000000000001, 0.898, 0.9005, 0.905, 0.913, 0.9215000000000001]
part_s_noniid=[0.653, 0.7645000000000001, 0.8059999999999999, 0.886, 0.9325, 0.948, 0.9545, 0.9740000000000001, 0.9790000000000001, 0.9890000000000001]

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
#n_converage_epoach=[1802,1101*(20/20*5),730,550,430,360]
#b_converage_epoach=[1600,800,533,400,320,266]
#k_converage_epoach=[1814,907,604,453,362,302]
#s_converage_epoach=[1886,943,662,521,437,381]

n_converage_epoach=[1190,600,397,299,249,208,168,137,118,108]
b_converage_epoach=[810,410,276,210,170,130,110,97,87,74]
k_converage_epoach=[917,463,312,236,181,150,125,110,95,85]
s_converage_epoach=[953,481,324,245,188,157,130,110,97,84]

batch_epoch=[10/35,20/45,30/55,40/65,50/75,60/85,70/95,80/105,90/115,100/125]

n_converage_epoach=np.array(n_converage_epoach)*batch_epoch
b_converage_epoach=np.array(b_converage_epoach)*batch_epoch
k_converage_epoach=np.array(k_converage_epoach)*batch_epoch
s_converage_epoach=np.array(s_converage_epoach)*batch_epoch




nt=np.array(new_york_time)*ratio*(100/23)
bt=np.array(beijing_time)*ratio*(100/23)
kt=np.array(KSR_time)*ratio*(100/23)
st=np.array(S_time)*ratio*(100/23)

nte=(nt/1200)*np.array(batch_ratio)+0.5*np.array([0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761,0.1134761])
bte=(bt/1200)*np.array(batch_ratio)+0.5*np.array([0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138,0.0224138])
kte=(kt/1200)*np.array(batch_ratio)+0.5*np.array([0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207,0.0348207])
ste=(st/1200)*np.array(batch_ratio)+0.5*np.array([0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457,0.0618457])

cut_ratio=[0.69,0.73,0.78,0.81,0.86,0.89,0.92,0.93,0.96,0.97]
#cut_ratio=[1,1,1,1,1,1]

nct=nte*n_converage_epoach*cut_ratio
bct=bte*b_converage_epoach*cut_ratio
kct=kte*k_converage_epoach*cut_ratio
sct=ste*s_converage_epoach*cut_ratio

print(list(nct),'nct')
print(list(bct),'bct')
print(list(kct),'kct')
print(list(sct),'sct')


ne=np.array(n_converage_epoach)*cut_ratio
be=np.array(b_converage_epoach)*cut_ratio
ke=np.array(k_converage_epoach)*cut_ratio
se=np.array(s_converage_epoach)*cut_ratio

print(list(ne),'ne')
print(list(be),'be')
print(list(ke),'ke')
print(list(se),'se')


new_york_acc=[43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917,43.38262917]
beijing_acc=[42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718,42.30507718]
KSR_acc=[44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634,44.0634]
s_acc=[52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642,52.94564642]

nc=np.array(new_york_acc)*np.array(part_n)
bc=np.array(beijing_acc)*np.array(part_b)
kc=np.array(KSR_acc)*np.array(part_k)
sc=np.array(s_acc)*np.array(part_s)



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
#plt.bar(index_male, height=nte, width=bar_width, color='y', label='New York')
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

bar_width = 0.15  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.tick_params(labelsize=Label_Size)
#plt.bar(index_male, height=nct, width=bar_width, color='y', label='New York')
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

bar_width = 0.15  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

#plt.bar(index_male, height=nc, width=bar_width, color='y', label='New York')
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

bar_width = 0.15  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

#plt.bar(index_male, height=ne, width=bar_width, color='y', label='New York')
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