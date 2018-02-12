import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

smooth = 10


def smoothResult(result):
	sum = 0
	smooth_result = []
	for i in range(len(result)):
		sum = sum + result[i]
		smooth_result.append(sum/(i+1))
	return smooth_result

def Smooth(result, smooth):
	return [sum(result[smooth*i:smooth*i+smooth])/smooth for i in range(0,int(600/smooth))]


#sns.set()
##########################  MinimaxQ-MinimaxQ ##########################
df = pd.read_csv('minimax-minimax_icml.csv')
#length = len(list(df.iloc[:, 0]))
minimax0 = list(df.iloc[:, 1])
minimax1 = list(df.iloc[:, 2])
minimax2 = list(df.iloc[:, 3])
minimax3 = list(df.iloc[:, 4])
minimax4 = list(df.iloc[:, 5])


minimax0 = Smooth(minimax0, smooth)
minimax1 = Smooth(minimax1, smooth)
minimax2 = Smooth(minimax2, smooth)
minimax3 = Smooth(minimax3, smooth)
minimax4 = Smooth(minimax4, smooth)

length = len(minimax3)

data_dict1 = {1: minimax0, 2:minimax1, 3:minimax2, 4:minimax3, 5:minimax4}
df = pd.DataFrame(data_dict1)
df = df.T
minimax_mean = list(df.mean())
minimax_std = list(df.std())

mini_list1 = [minimax_mean[i]+minimax_std[i] for i in range(length)]
mini_list2 = [minimax_mean[i]-minimax_std[i] for i in range(length)]



##################################   SPQ (PR = 10)   ############################################
df = pd.read_csv('SPQ_1_10_icml.csv')


spq0 = list(df.iloc[:, 1])
spq1 = list(df.iloc[:, 2])
spq2 = list(df.iloc[:, 3])
spq3 = list(df.iloc[:, 4])
# spq4 = list(df.iloc[:, 5])


spq0 = Smooth(spq0, smooth)
spq1 = Smooth(spq1, smooth)
spq2 = Smooth(spq2, smooth)
spq3 = Smooth(spq3, smooth)
#spq4 = Smooth(spq4, smooth)


data_dict2 = {1: spq0, 2: spq1, 3:spq2, 4:spq3} #, 4:spq3, 5:spq4
df = pd.DataFrame(data_dict2)
df = df.T
spq_mean = list(df.mean())
spq_std = list(df.std())

spq_list1 = [spq_mean[i]+ spq_std[i] for i in range(length)]
spq_list2 = [spq_mean[i]- spq_std[i] for i in range(length)]

######################################  HAMMQ (4, 4) Heuristic   ########################################

df = pd.read_csv('HAMMQ_(4, 4)_icml.csv')


hammq_1_1 = list(df.iloc[:, 1])
hammq_1_2 = list(df.iloc[:, 2])
hammq_1_3 = list(df.iloc[:, 3])
hammq_1_4 = list(df.iloc[:, 4])
hammq_1_5 = list(df.iloc[:, 5])


hammq_1_1 = Smooth(hammq_1_1, smooth)
hammq_1_2 = Smooth(hammq_1_2, smooth)
hammq_1_3 = Smooth(hammq_1_3, smooth)
hammq_1_4 = Smooth(hammq_1_4, smooth)
hammq_1_5 = Smooth(hammq_1_5, smooth)


data_dict3 = {1: hammq_1_1, 2:hammq_1_2, 3:hammq_1_3, 4:hammq_1_4, 5:hammq_1_5}
df = pd.DataFrame(data_dict3)
df = df.T
hammq_mean = list(df.mean())
hammq_std = list(df.std())

hammq_list1 = [hammq_mean[i]+ hammq_std[i] for i in range(length)]
hammq_list2 = [hammq_mean[i]- hammq_std[i] for i in range(length)]






####################################  HAMMQ (6, 4) Heuristic ######################################################


df = pd.read_csv('HAMMQ_(6, 4)_icml.csv')


hammq_2_1 = list(df.iloc[:, 1])
hammq_2_2 = list(df.iloc[:, 2])
hammq_2_3 = list(df.iloc[:, 3])
hammq_2_4 = list(df.iloc[:, 4])
hammq_2_5 = list(df.iloc[:, 5])


hammq_2_1 = Smooth(hammq_2_1, smooth)
hammq_2_2 = Smooth(hammq_2_2, smooth)
hammq_2_3 = Smooth(hammq_2_3, smooth)
hammq_2_4 = Smooth(hammq_2_4, smooth)
hammq_2_5 = Smooth(hammq_2_5, smooth)


data_dict4 = {1: hammq_2_1, 2:hammq_2_2, 3:hammq_2_3, 4:hammq_2_4, 5:hammq_2_5}
df = pd.DataFrame(data_dict4)
df = df.T
hammq_mean_2 = list(df.mean())
hammq_std_2 = list(df.std())

hammq_list1_2 = [hammq_mean_2[i]+ hammq_std_2[i] for i in range(length)]
hammq_list2_2 = [hammq_mean_2[i]- hammq_std_2[i] for i in range(length)]



####################################  HAMMQ (9, 4) Heuristic ######################################################

df = pd.read_csv('HAMMQ_(9, 4)_icml.csv')


hammq_3_1 = list(df.iloc[:, 1])
hammq_3_2 = list(df.iloc[:, 2])
hammq_3_3 = list(df.iloc[:, 3])
hammq_3_4 = list(df.iloc[:, 4])
hammq_3_5 = list(df.iloc[:, 5])


hammq_3_1 = Smooth(hammq_3_1, smooth)
hammq_3_2 = Smooth(hammq_3_2, smooth)
hammq_3_3 = Smooth(hammq_3_3, smooth)
hammq_3_4 = Smooth(hammq_3_4, smooth)
hammq_3_5 = Smooth(hammq_3_5, smooth)


data_dict5 = {1: hammq_3_1, 2:hammq_3_2, 3:hammq_3_3, 4:hammq_3_4, 5:hammq_3_5}
df = pd.DataFrame(data_dict5)
df = df.T
hammq_mean_3 = list(df.mean())
hammq_std_3 = list(df.std())

hammq_list1_3 = [hammq_mean_3[i]+ hammq_std_3[i] for i in range(length)]
hammq_list2_3 = [hammq_mean_3[i]- hammq_std_3[i] for i in range(length)]




##########################################   None-overlap   ##############################################################################
df = pd.read_csv('HAMMQ_(6, 4)_icml.csv')
spq_two = list(df.iloc[:, 3])
spq_two = [sum(spq_two[10*i:10*i+10])/10 for i in range(0,int(300/5))]
df = pd.read_csv('SPQ0_icml.csv')
spq_non = list(df.iloc[:, 1])
spq_non = [sum(spq_non[5*i:5*i+5])/5 for i in range(0,int(300/5))]
df = pd.read_csv('SPQ3_icml.csv')
spq_three = list(df.iloc[:, 1])
spq_three = [sum(spq_three[5*i:5*i+5])/5 for i in range(0,int(300/5))]
print('three = ', spq_three)



########################################################################################################################
########################################################################################################################
########################################################################################################################


# plt.plot(minimax_mean, marker="*", linestyle='--', label = 'Baseline(MinimaxQ)', color = 'green', linewidth=1)
# l = plt.fill_between(np.arange(length), mini_list1, mini_list2)
# # l.set_edgecolors([0, 0, .5, .3])
# # l.set_facecolors([[.9,.7,.8,.5]])
# l.set_facecolor('green')
# l.set_alpha(0.2)


plt.plot(spq_two, marker="X", linestyle='--', label = 'adjacent neighbor overlap', color = 'red', linewidth=1)
# l = plt.fill_between(np.arange(length), spq_list1, spq_list2)
# l.set_facecolor('red')
# l.set_alpha(0.2)
plt.plot(spq_three, marker="*", linestyle='--', label = 'All agent overlap', color = 'blue', linewidth=1)
plt.plot(spq_non, marker="^", linestyle='--', label = 'None overlap', color = 'green', linewidth=1)


# plt.plot(hammq_mean, marker="^", linestyle='--', label = 'HAMMQ ((4, 6) Heuristic)', color = 'blue', linewidth=1)
# l = plt.fill_between(np.arange(length), hammq_list1, hammq_list2)
# l.set_facecolor('blue')
# l.set_alpha(0.2)


# plt.plot(hammq_mean_2, marker="D", linestyle='--', label = 'HAMMQ ((6, 6) Heuristic)', color = 'blue', linewidth=1)
# l = plt.fill_between(np.arange(length), hammq_list1_2, hammq_list2_2)
# l.set_facecolor('blue')
# l.set_alpha(0.2)


# plt.plot(hammq_mean_3, marker="D", linestyle='--', label = 'HAMMQ ((9, 6) Heuristic)', color = 'blue', linewidth=1)
# l = plt.fill_between(np.arange(length), hammq_list1_3, hammq_list2_3)
# l.set_facecolor('blue')
# l.set_alpha(0.2)



plt.title('SPQ performance comparison based on partition overlap')
plt.xlabel('Game (grouped by each 10 games)')
plt.ylabel('player win times')
plt.legend(loc = 2)#bbox_to_anchor=(1.04,1), loc = 2
plt.savefig("overlap.pdf", bbox_inches="tight")

plt.show()



