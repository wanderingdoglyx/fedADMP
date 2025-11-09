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

power=5

iid_n=[4.15922207764263, 5.3137951104900685, 8.105604697271799, 9.832126994372755, 11.246104684473403, 12.442188200270971, 14.822614634497823, 17.051693847270098, 18.855597657281873, 20.954125272475753]
iid_b=[0.5969324900086499, 0.7633303783351164, 1.160409917499851, 1.4123919086826244, 1.6155107912701765, 1.7861081704017256, 2.125888371563228, 2.458427464883203, 2.686417969253283, 3.0100747280120848]
iid_k=[1.051393330539439, 1.3429920979733612, 2.0467247365925694, 2.479455765206934, 2.8391684207416383, 3.1503279310707697, 3.737214417489412, 4.315772494642242, 4.742568494269631, 5.260808584708434]
iid_s=[1.9415176336581896, 2.480092496217881, 3.7796697900953977, 4.579177769050438, 5.237718409152824, 5.817684760963646, 6.895008717032292, 7.936658192889027, 8.760293707806714, 9.759095328257635]

noniid_n=[8.990870612887791, 11.879881077758931, 13.917759027796826, 15.72088754715216, 17.678535772941142, 18.366156735904916, 19.581656511721583, 19.376924826443293, 20.041078396318333, 21.20942389341501]
noniid_b=[1.2087882922675162, 1.6034489561392542, 1.9111656356961386, 2.1808992707599346, 2.384000299617448, 2.2672987223144645, 2.532463947064707, 2.7098575465189856, 2.9185609805823627, 2.8704320859909056]
noniid_k=[2.125970637496506, 2.813033951198425, 3.3563445696251533, 3.8075973489643182, 3.9432894732522756, 4.0642400418045534, 4.470779999867705, 4.774084617967081, 4.951033043468296, 5.1222076712510525]
noniid_s=[3.9242127356866483, 5.190518879663343, 6.190541967399195, 7.0206417610598075, 7.274608901601144, 7.555434754498241, 8.2582562485185, 8.47933567616349, 8.978745664172138, 8.990612059373124]

iid_batch_random_n=[16.653063111610972, 16.893988336262115, 17.602068113152797, 18.36621905288563, 20.412679920874663, 21.149430519602, 21.653638141170923, 20.951097523419577, 21.70735858217627, 22.833244140385908]
iid_batch_random_b=[2.238944006533439, 2.2802149226029984, 2.4170921049142615, 2.547878646819105, 2.752709789603807, 2.6108954881777904, 2.8004319624544127, 2.93000702636967, 3.1612214007384796, 3.090197595863172]
iid_batch_random_k=[3.9377632481870624, 4.000322765818276, 4.24483576310365, 4.44829606942082, 4.553154291324433, 4.680149413758529, 4.943843624922829, 5.161928893584874, 5.362677764811032, 5.514370158632985]
iid_batch_random_s=[7.26850608346083, 7.38126986704169, 7.8293056036350706, 8.201998385632098, 8.399695331139664, 8.700415387128517, 9.132084481635792, 9.168195371258895, 9.7252698178546, 9.678946610660818]

noniid_batch_random_n=[24.13487407479851, 23.14244977570153, 22.566753991221532, 22.674344509735338, 23.735674326598446, 23.763405078204492, 23.536563196924916, 22.528061853139327, 22.611831856433614, 23.53942694885145]
noniid_batch_random_b=[3.244846386280347, 3.123582085757532, 3.098836031941361, 3.1455291936038328, 3.200825336748613, 2.933590436154821, 3.0439477852765355, 3.1505451896448067, 3.292938959102583, 3.1857707173847136]
noniid_batch_random_k=[5.70690325824212, 5.479894199751064, 5.442097132184166, 5.49172354249484, 5.2943654550284105, 5.258594846919696, 5.373743070568293, 5.550461175897714, 5.586122671678158, 5.684917689312355]
noniid_batch_random_s=[10.534066787624392, 10.111328584988616, 10.037571286711628, 10.125923932879134, 9.767087594348448, 9.7757476259871, 9.92617878438673, 9.8582745927515, 10.130489393598543, 9.97829547490806]

iid_batch_n=[15.237361661333098, 15.234530345296623, 16.019642039313155, 16.973910485841635, 19.11698254316622, 19.95572339054377, 20.890276488215903, 20.43164840004328, 21.396299427345028, 22.644790474063093]
iid_batch_b=[2.0486078940573247, 2.056234248493553, 2.1997950078752897, 2.354728163287991, 2.577980532805076, 2.4635318105830235, 2.7017069041529425, 2.857361500076734, 3.1159214798555404, 3.064692089435807]
iid_batch_k=[3.603008209606394, 3.6073802150506524, 3.8632250503325802, 4.111079169351905, 4.264142334432576, 4.415994494275744, 4.769557026218309, 5.03394715967651, 5.2858324719438645, 5.468857504359582]
iid_batch_s=[6.65059943607571, 6.656224095175961, 7.125450340560102, 7.580219560105983, 7.866522849519554, 8.209349444166318, 8.810147764171525, 8.94088375108244, 9.585909158431688, 9.599060878392983]

noniid_batch_n=[22.083132842511738, 20.869219651091267, 20.538002614504045, 20.955445044248936, 22.22904946879793, 22.422161112970528, 22.706822269799897, 21.969514408648685, 22.2878119034844, 23.345144818621748]
noniid_batch_b=[2.9689969479091665, 2.816759244511717, 2.820250010096525, 2.9070718065283843, 2.9976517823314834, 2.768013270318004, 2.9366379392966766, 3.072431720512617, 3.2457515415161877, 3.159476380861657]
noniid_batch_k=[5.221751028415064, 4.941616732946099, 4.952852628631513, 5.075406381915933, 4.95830504003788, 4.96179156660196, 5.1843011154546845, 5.41284640825431, 5.506075491608192, 5.637997427174827]
noniid_batch_s=[9.6385499073561, 9.11811519887118, 9.135192744307822, 9.358295753217263, 9.147119592464598, 9.223988139512718, 9.576247569751658, 9.613853495787568, 9.98532204003301, 9.895939049889675]





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
#plt.bar(index_male, height=np.array(iid_n)*5, width=bar_width, color='y', label='New York')
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
#plt.bar(index_male, height=np.array(noniid_n)*5, width=bar_width, color='y', label='New York')
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

#plt.bar(index_male, height=np.array(iid_batch_random_n)*5, width=bar_width, color='y', label='New York')
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

#plt.bar(index_male, height=np.array(noniid_batch_random_n)*5, width=bar_width, color='y', label='New York')
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

#plt.bar(index_male, height=np.array(iid_batch_n)*5, width=bar_width, color='y', label='New York')
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

#plt.bar(index_male, height=np.array(noniid_batch_n)*5, width=bar_width, color='y', label='New York')
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