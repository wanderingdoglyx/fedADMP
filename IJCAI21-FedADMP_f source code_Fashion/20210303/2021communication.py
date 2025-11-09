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


def LSTMcell_cost(input_size,hidden_size,batch_size=1):
   LSTM_w1=np.random.random((input_size,hidden_size))
   LSTM_w1_c = 3*sys.getsizeof(LSTM_w1)
   LSTM_w2=np.random.random((hidden_size,hidden_size))
   LSTM_w2_c=3*sys.getsizeof(LSTM_w2)
   LSTM_b1=np.random.random((batch_size,hidden_size))
   LSTM_b1_c =4*sys.getsizeof(LSTM_b1)
   LSTM_cost=LSTM_w1_c+LSTM_w2_c+LSTM_b1_c
   return LSTM_cost

def Linear_cost(input_size,hidden_size,batch_size=1):
    Linear_w=np.random.random((input_size,hidden_size))
    Linear_w_c= sys.getsizeof(Linear_w)
    Linear_b=np.random.random((hidden_size,batch_size))
    Linear_b_c= sys.getsizeof(Linear_b)
    Linear_total_cost=Linear_b_c+Linear_w_c
    return Linear_total_cost

batch10_n=48352
batch10_b= 34016
batch10_k=37088
batch10_s=39136

batch35_n=73952
batch35_b= 59616
batch35_k=62688
batch35_s=64736

batch55_n=94432
batch55_b= 80096
batch55_k=83168
batch55_s=85216


client_num=np.array([10,20,30,40,50,60,70,80,90,100])

ne_noniid_batch_com=np.array([340.0, 266.66666666666663, 216.54545454545453, 184.0, 166.0, 146.82352941176472, 123.78947368421052, 104.38095238095238, 92.34782608695653, 86.4])*73952.*client_num
be_noniid_batch_com=np.array([231.42857142857142, 182.2222222222222, 150.54545454545453, 129.23076923076923, 113.33333333333333, 91.76470588235294, 81.05263157894737, 73.9047619047619, 68.08695652173914, 59.2])*59616.*client_num
ke_noniid_batch_com=np.array([262.0, 205.77777777777777, 170.18181818181816, 145.23076923076923, 120.66666666666666, 105.88235294117648, 92.10526315789473, 83.80952380952381, 74.34782608695653, 68.0])*62688.*client_num
se_noniid_batch_com= np.array([272.2857142857143, 213.77777777777777, 176.72727272727272, 150.76923076923077, 125.33333333333333, 110.82352941176471, 95.78947368421052, 83.80952380952381, 75.91304347826087, 67.2])*64736.*client_num

ne_iid_batch_com=np.array([234.6, 194.66666666666663, 168.90545454545455, 149.04000000000002, 142.76, 130.6729411764706, 113.88631578947368, 97.07428571428572, 88.65391304347827, 83.808])*73952.*client_num
be_iid_batch_com=np.array([159.68571428571425, 133.0222222222222, 117.42545454545454, 104.67692307692307, 97.46666666666665, 81.67058823529412, 74.56842105263158, 68.73142857142857, 65.36347826086957, 57.424])*59616.*client_num
ke_iid_batch_com=np.array([180.77999999999997, 150.21777777777777, 132.74181818181816, 117.63692307692308, 103.77333333333333, 94.23529411764707, 84.73684210526315, 77.94285714285715, 71.37391304347827, 65.96])*62688.*client_num
se_iid_batch_com= np.array([187.87714285714284, 156.05777777777777, 137.84727272727272, 122.12307692307694, 107.78666666666666, 98.6329411764706, 88.12631578947368, 77.94285714285715, 72.87652173913044, 65.184])*64736.*client_num

ne_noniid_com=np.array([1190, 600, 397, 299, 249, 208, 168, 137, 118, 108])*48352.*client_num
be_noniid_com=np.array([810, 410, 276, 210, 170, 130, 110, 97, 87, 74])*34016.*client_num
ke_noniid_com=np.array([917, 463, 312, 236, 181, 150, 125, 110, 95, 85])*37088.*client_num
se_noniid_com=np.array([953, 481, 324, 245, 188, 157, 130, 110, 97, 84])*39136.*client_num

ne_iid_com=np.array([550.5, 319.0, 231.21, 187.0, 158.4, 140.91, 127.17, 120.56, 111.02000000000001, 106.7])*48352.*client_num
be_iid_com=np.array([400.0, 231.99999999999997, 167.58, 136.0, 115.19999999999999, 102.41, 92.34, 88.0, 80.08, 77.6])*34016.*client_num
ke_iid_com=np.array([453.5, 262.74, 190.26, 153.68, 130.32, 116.27, 104.49000000000001, 99.44, 91.0, 87.3])*37088.*client_num
se_iid_com= np.array([471.5, 273.18, 197.82, 159.8, 135.35999999999999, 120.89, 108.54, 102.96, 94.64, 91.17999999999999])*39136.*client_num

ne_noniid_batch_com_random=np.array([198.33333333333331, 171.42857142857142, 148.875, 132.88888888888889, 124.5, 113.45454545454544, 98.0, 84.3076923076923, 75.85714285714286, 72.0])*94432.*client_num
be_noniid_batch_com_random=np.array([135.0, 117.14285714285714, 103.5, 93.33333333333333, 85.0, 70.9090909090909, 64.16666666666667, 59.69230769230769, 55.92857142857143, 49.33333333333333])*80096.*client_num
ke_noniid_batch_com_random=np.array([152.83333333333331, 132.28571428571428, 117.0, 104.88888888888889, 90.5, 81.81818181818181, 72.91666666666667, 67.6923076923077, 61.07142857142858, 56.666666666666664] )*83168.*client_num
se_noniid_batch_com_random=np.array([158.83333333333331, 137.42857142857142, 121.5, 108.88888888888889, 94.0, 85.63636363636363, 75.83333333333334, 67.6923076923077, 62.35714285714286, 56.0])*85216.*client_num

ne_iid_batch_com_random=np.array([136.84999999999997, 125.14285714285712, 116.1225, 107.64, 107.07, 100.97454545454545, 90.16000000000001, 78.40615384615386, 72.82285714285715, 69.84])*94432.*client_num
be_iid_batch_com_random=np.array([93.14999999999999, 85.5142857142857, 80.73, 75.6, 73.1, 63.10909090909091, 59.03333333333334, 55.51384615384616, 53.691428571428574, 47.853333333333325])*80096.*client_num
ke_iid_batch_com_random=np.array([105.45499999999998, 96.56857142857142, 91.26, 84.96000000000001, 77.83, 72.81818181818181, 67.08333333333334, 62.95384615384616, 58.62857142857143, 54.96666666666666])*83168.*client_num
se_iid_batch_com_random= np.array([109.59499999999998, 100.32285714285713, 94.77000000000001, 88.2, 80.84, 76.21636363636362, 69.76666666666668, 62.95384615384616, 59.862857142857145, 54.32])*85216.*client_num








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
plt.bar(index_male, height=ne_iid_com/10e6, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be_iid_com/10e6, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke_iid_com/10e6, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se_iid_com/10e6, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB ',font1)  # 纵坐标轴标题
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
plt.bar(index_male, height=ne_noniid_com/10e6, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be_noniid_com/10e6, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke_noniid_com/10e6, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se_noniid_com/10e6, width=bar_width, color='r', label='Seattle')

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6,prop = font3)# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), ncol=4,prop = font3)
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB ',font1)  # 纵坐标轴标题
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

plt.bar(index_male, height=ne_iid_batch_com_random/10e6, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be_iid_batch_com_random/10e6, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke_iid_batch_com_random/10e6, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se_iid_batch_com_random/10e6, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB   ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(c)',font1)


plt.subplot(2,3,4)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=ne_noniid_batch_com_random/10e6, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be_noniid_batch_com_random/10e6, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke_noniid_batch_com_random/10e6, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se_noniid_batch_com_random/10e6, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(d)',font1)


plt.subplot(2,3,5)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=ne_iid_batch_com/10e6, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be_iid_batch_com/10e6, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke_iid_batch_com/10e6, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se_iid_batch_com/10e6, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB   ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(e)',font1)


plt.subplot(2,3,6)
waters = ( '10','20','30','40','50','60','70','80','90','100')

bar_width = 0.15
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标
index_female1 = index_female + bar_width
index_female2 = index_female1 + bar_width

plt.bar(index_male, height=ne_noniid_batch_com/10e6, width=bar_width, color='y', label='New York')
plt.bar(index_female, height=be_noniid_batch_com/10e6, width=bar_width, color='b', label='Beijing')
plt.bar(index_female1, height=ke_noniid_batch_com/10e6, width=bar_width, color='g', label='KSP')
plt.bar(index_female2, height=se_noniid_batch_com/10e6, width=bar_width, color='r', label='Seattle')

plt.tick_params(labelsize=Label_Size)
#plt.legend(loc='upper center', bbox_to_anchor=(0, 1.18), ncol=4,prop = font3) # 显示图例
plt.xticks(index_male + bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('MB ',font1)  # 纵坐标轴标题
plt.xlabel('Selected number',font1)
plt.title('(f)',font1)
plt.show()

