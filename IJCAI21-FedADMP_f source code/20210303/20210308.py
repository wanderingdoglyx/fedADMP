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

ne_noniid_batch=[340.0, 266.66666666666663, 216.54545454545453, 184.0, 166.0, 146.82352941176472, 123.78947368421052, 104.38095238095238, 92.34782608695653, 86.4]
be_noniid_batch=[231.42857142857142, 182.2222222222222, 150.54545454545453, 129.23076923076923, 113.33333333333333, 91.76470588235294, 81.05263157894737, 73.9047619047619, 68.08695652173914, 59.2]
ke_noniid_batch=[262.0, 205.77777777777777, 170.18181818181816, 145.23076923076923, 120.66666666666666, 105.88235294117648, 92.10526315789473, 83.80952380952381, 74.34782608695653, 68.0]
se_noniid_batch= [272.2857142857143, 213.77777777777777, 176.72727272727272, 150.76923076923077, 125.33333333333333, 110.82352941176471, 95.78947368421052, 83.80952380952381, 75.91304347826087, 67.2]

ne_iid_batch=[234.6, 194.66666666666663, 168.90545454545455, 149.04000000000002, 142.76, 130.6729411764706, 113.88631578947368, 97.07428571428572, 88.65391304347827, 83.808]
be_iid_batch=[159.68571428571425, 133.0222222222222, 117.42545454545454, 104.67692307692307, 97.46666666666665, 81.67058823529412, 74.56842105263158, 68.73142857142857, 65.36347826086957, 57.424]
ke_iid_batch=[180.77999999999997, 150.21777777777777, 132.74181818181816, 117.63692307692308, 103.77333333333333, 94.23529411764707, 84.73684210526315, 77.94285714285715, 71.37391304347827, 65.96]
se_iid_batch= [187.87714285714284, 156.05777777777777, 137.84727272727272, 122.12307692307694, 107.78666666666666, 98.6329411764706, 88.12631578947368, 77.94285714285715, 72.87652173913044, 65.184]

ne_noniid=[1190, 600, 397, 299, 249, 208, 168, 137, 118, 108]
be_noniid=[810, 410, 276, 210, 170, 130, 110, 97, 87, 74]
ke_noniid=[917, 463, 312, 236, 181, 150, 125, 110, 95, 85]
se_noniid=[953, 481, 324, 245, 188, 157, 130, 110, 97, 84]

ne_iid=[550.5, 319.0, 231.21, 187.0, 158.4, 140.91, 127.17, 120.56, 111.02000000000001, 106.7]
be_iid=[400.0, 231.99999999999997, 167.58, 136.0, 115.19999999999999, 102.41, 92.34, 88.0, 80.08, 77.6]
ke_iid=[453.5, 262.74, 190.26, 153.68, 130.32, 116.27, 104.49000000000001, 99.44, 91.0, 87.3]
se_iid= [471.5, 273.18, 197.82, 159.8, 135.35999999999999, 120.89, 108.54, 102.96, 94.64, 91.17999999999999]



















x1=[10,20,30,40,50,60,70,80,90,100]


plt.subplot(2,3,1)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.tick_params(labelsize=Label_Size)
plt.bar(index_male, height=np.array(iid_n)*5, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=np.array(iid_b)*5, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=np.array(iid_k)*5, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=np.array(iid_s)*5, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('')  # 图形标题
plt.title('(a)',font1)
plt.subplot(2,3,2)

waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width
index_female3 = index_female2 + bar_width
# 使用两次 bar 函数画出两组条形图
plt.tick_params(labelsize=Label_Size)
plt.bar(index_male, height=np.array(noniid_n)*5, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=np.array(noniid_b)*5, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=np.array(noniid_k)*5, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=np.array(noniid_s)*5, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), ncol=4,prop = font3)
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
#plt.title('购买饮用水情况的调查结果')  # 图形标题
plt.title('(b)',font1)
#############################s t
plt.subplot(2,3,3)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=np.array(iid_batch_random_n)*5, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=np.array(iid_batch_random_b)*5, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=np.array(iid_batch_random_k)*5, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=np.array(iid_batch_random_s)*5, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J   ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(c)',font1)


plt.subplot(2,3,4)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=np.array(noniid_batch_random_n)*5, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=np.array(noniid_batch_random_b)*5, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=np.array(noniid_batch_random_k)*5, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=np.array(noniid_batch_random_s)*5, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(d)',font1)


plt.subplot(2,3,5)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=np.array(iid_batch_n)*5, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=np.array(iid_batch_b)*5, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=np.array(iid_batch_k)*5, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=np.array(iid_batch_s)*5, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(e)',font1)


plt.subplot(2,3,6)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=np.array(noniid_batch_n)*5, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=np.array(noniid_batch_b)*5, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=np.array(noniid_batch_k)*5, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=np.array(noniid_batch_s)*5, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('J ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(f)',font1)
plt.show()

