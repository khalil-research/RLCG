import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Parameters
from copy import deepcopy



##############################################################
##############################################################
###########				Traing functions		############## data stored in the form of [step1, step2, step3,....] 
############################################################## len = # of train instance 
##############################################################

## get the training total steps data
def plot_training(model_index): #### input: the model number; use training data stored in folders corresponds to that model; 
								#### output: two plots

	train_data_path = 'save_data/training_data/'
	steps = np.load(path_data+"model_"+str(model_index)+"_total_steps")

	num_episodes= []
	for i in range(len(steps)):
	    num_episodes.append(i)

	step_curr = deepcopy(steps)
	step_curr = step_curr[0:120]
	num_episodes_curr = deepcopy(num_episodes)
	num_episodes_curr = num_episodes[0:120]


	num_episodes_1 = num_episodes_curr[0:40]
	num_episodes_2 = num_episodes_curr[41:80]	
	num_episodes_3 = num_episodes_curr[81:120]

	step_curr_1 = step_curr[0:40]
	step_curr_2 = step_curr[41:80]
	step_curr_3 = step_curr[81:120]


	m1, b1 = np.polyfit(num_episodes_1, step_curr_1, 1)
	m2, b2 = np.polyfit(num_episodes_2, step_curr_2, 1)
	m3, b3 = np.polyfit(num_episodes_3, step_curr_3, 1)

	# fig, axs = plt.subplots(2)
	            

	fig, ax = plt.subplots(2)
	fig.suptitle('Training plots for model '+str(model_index))
	ax[0].plot(num_episodes, steps)
	ax[0].set_xlabel("Episodes")
	ax[0].set_ylabel("Steps till convergence")
	ax[0].legend()
	ax[0].grid(True, linestyle='-.')
	ax[0].tick_params(labelcolor='black', labelsize='medium', width=1)

	ax[1].plot(num_episodes_curr, step_curr)
	ax[1].plot(num_episodes_1, m1*np.asarray(num_episodes_1)+b1)
	ax[1].plot(num_episodes_2, m1*np.asarray(num_episodes_2)+b2)
	ax[1].plot(num_episodes_3, m1*np.asarray(num_episodes_3)+b3)
	ax[1].set_xlabel("Episodes")
	ax[1].set_ylabel("Steps till convergence")
	ax[1].legend()
	ax[1].grid(True, linestyle='-.')
	ax[1].tick_params(labelcolor='black', labelsize='medium', width=1)

	plt.show() 
	plt.savefig("save_graph/training_plots/model_"+str(model_index)+".png")  


################## Validation
def plot_val_model_comparison():


	total_results = np.load("save_data/validation_data/model_compare")
	for i in range(len(total_results)):

		result_each_model = total_results[i]
		plt.plot(list(range(len(result_each_model))), result_each_model,c='r',label="model"+str(i))

	plt.xlabel('instances i')
	plt.ylabel('total steps')
	plt.legend()
	plt.show()


############################################################## Have this data for each prob_size n
############################################################## data stored in testinng_data folder named after model in the form of:
############################################################## (Greedy,RL,Expert,True obj), first 3 elements are
############################################################## list of info about each testing instance: 
###########				Testing functions		##############  [(historical obj list, step, time, obj), (historical obj list, step, time, obj)]..
##############################################################
############################################################## All four elements in (Greedy,RL,Expert,True obj) have the same length = # of testing instances

# this is how data is saved
# complete_data = (Greedy,RL,Expert)
# path = 'save_data/testing_data/'
# np.save(path+'testing_result_model_'+str(model_index)+"_size"+str(prob_size),complete_data)

# def plot_testing_results(model_index,prob_size):
# 	path = 'save_data/testing_data/'
# 	DATA = np.load(path+'testing_result_model_'+str(model_index)+"_size"+str(prob_size))
# 	# DATA = (Greedy,RL,Expert), each element here is a list of tuple
# 	# [(historical obj list, step, time, obj), (historical obj list, step, time, obj), .... ]



## turn this into a shaded version of comparison plots 
def compare_trajectory(hist1,hist2,hist3=None):
    if hist3!=None:
        len1 = len(hist1)
        len2 = len(hist2)
        len3 = len(hist3)

        max_len = max(len1,len2,len3)


        if len(hist1) != max_len:
            diff = max_len-len(hist1)
            for i in range(diff):
                hist1.append(None)
        if len(hist2) != max_len:
            diff = max_len-len(hist1)
            for i in range(diff):
                hist2.append(None)


        if len(hist3) != max_len:
            diff = max_len-len(hist1)
            for i in range(diff):
                hist3.append(None)
          

        plt.plot(list(range(len(hist1))), hist1,c='r',label="greedy")
        plt.scatter(list(range(len(hist1))), hist1, c='r')
        plt.plot(list(range(len(hist2))), hist2,c='g',label='RL')
        plt.scatter(list(range(len(hist2))), hist2, c='g')

        plt.plot(list(range(len(hist3))), hist3,c='blue',label='expert')
        plt.scatter(list(range(len(hist3))), hist3, c='blue')

        plt.xlabel('history')
        plt.ylabel('objective function value')
        plt.legend()
        plt.show()
    
    else:
        len1 = len(hist1)
        len2 = len(hist2)


        max_len = max(len1,len2)


        if len(hist1) != max_len:
            diff = max_len-len(hist1)
            for i in range(diff):
                hist1.append(None)
        if len(hist2) != max_len:
            diff = max_len-len(hist1)
            for i in range(diff):
                hist2.append(None)

        plt.plot(list(range(len(hist1))), hist1,c='r',label="greedy")
        plt.scatter(list(range(len(hist1))), hist1, c='r')
        plt.plot(list(range(len(hist2))), hist2,c='g',label='RL')
        plt.scatter(list(range(len(hist2))), hist2, c='g')


        plt.xlabel('history')
        plt.ylabel('objective function value')
        plt.legend()
        plt.show()
    


   
      













# ##### read data from data file
# G50 = np.load("/Greedy50.npy")
# RL50 = np.load("/RL50.npy")
# # G100 = np.load("/Greedy100.npy")
# # RL100 = np.load("/RL100.npy")
# G200 = np.load("/Greedy200.npy")
# RL200 = np.load("/RL200.npy")
# G500 = np.load("/Greedy500.npy")
# RL500 = np.load("/RL500.npy")
# G750_1 = np.load("/Greedy750_1.npy")
# RL750_1 = np.load("/RL750_1.npy")
# G750_2 = np.load("/Greedy750_2.npy")
# RL750_2 = np.load("/RL750_2.npy")
# G750 = np.concatenate([G750_1, G750_2])
# RL750 = np.concatenate([RL750_1, RL750_2])



# #### functions for generating tables
# def stat(arr):
#   return [arr[:, 0].mean().round(1), arr[:, 0].std().round(1)], [arr[:, 1].mean().round(1), arr[:, 1].std().round(1)], [arr[:, 2].mean().round(1), arr[:, 2].std().round(1)]

# def table_it(G, RL):
#   return pd.DataFrame(data = np.asarray([stat(G)[0],stat(RL)[0]]), columns = ["mean", "std"])

# def table_time(G, RL):
#   return pd.DataFrame(data = np.asarray([stat(G)[1], stat(RL)[1]]), columns = ["mean", "std"])

# def table_obj(G, RL):
#   return pd.DataFrame(data = np.asarray([stat(G)[2], stat(RL)[2]]), columns = ["mean", "std"])



# #### functions for scatter plot
# sns.set_style('whitegrid')
# sns.set(rc={'figure.figsize':(20, 10)})

# def scatter(G50, RL50, G200, RL200, G500, RL500, G750, RL750):
#   # plt.scatter(arr1[:, 0], arr2[:, 0])
#   # plt.xlabel("Random (n = {})".format(n))
#   # plt.ylabel("Greedy (n = {})".format(n))
#   # low_x, high_x = plt.get_xlim()
#   # low_y, high_y = plt.get_ylim()
#   # plt.show()

#   i = 0
#   fig = plt.figure()
#   grl50 = fig.add_subplot(341)  
#   grl50.set_xlabel('Iterations for Greedy (n = 50)')
#   grl50.set_ylabel('Iterations for RL (n = 50)', fontweight='bold')
#   plt.scatter(G50[:, i], RL50[:, i])
#   plt.plot([0, 80], [0, 80], "r")
#   grl200 = fig.add_subplot(342)
#   grl200.set_xlabel('Iterations for Greedy (n = 200)')
#   grl200.set_ylabel('Iterations for RL (n = 200)', fontweight='bold')
#   plt.scatter(G200[:, i], RL200[:, i])  
#   plt.plot([0, 150], [0, 150], "r")
#   grl500 = fig.add_subplot(343)
#   grl500.set_xlabel('Iterations for Greedy (n = 500)')
#   grl500.set_ylabel('Iterations for RL (n = 500)', fontweight='bold')
#   plt.scatter(G500[:, i], RL500[:, i])  
#   plt.plot([0, 300], [0, 300], "r")
#   grl750 = fig.add_subplot(344)
#   grl750.set_xlabel('Iterations for Greedy (n = 750)')
#   grl750.set_ylabel('Iterations for RL (n = 750)', fontweight='bold')
#   plt.scatter(G750[:, i], RL750[:, i])  
#   plt.plot([0, 400], [0, 400], "r")

#   i = 1
#   grl50 = fig.add_subplot(345)  
#   grl50.set_xlabel('Time(s) for Greedy (n = 50)')
#   grl50.set_ylabel('Time(s) for RL (n = 50)', fontweight='bold')
#   plt.scatter(G50[:, i], RL50[:, i])
#   plt.plot([0, 15], [0, 15], "r")
#   grl200 = fig.add_subplot(346)
#   grl200.set_xlabel('Time(s) for Greedy (n = 200)')
#   grl200.set_ylabel('Time(s) for RL (n = 200)', fontweight='bold')
#   plt.scatter(G200[:, i], RL200[:, i])  
#   plt.plot([0, 70], [0, 70], "r")
#   grl500 = fig.add_subplot(347)
#   grl500.set_xlabel('Time(s) for Greedy (n = 500)')
#   grl500.set_ylabel('Time(s) for RL (n = 500)', fontweight='bold')
#   plt.scatter(G500[:, i], RL500[:, i])  
#   plt.plot([0, 100], [0, 100], "r")
#   grl750 = fig.add_subplot(348)
#   grl750.set_xlabel('Time(s) for Greedy (n = 750)')
#   grl750.set_ylabel('Time(s) for RL (n = 750)', fontweight='bold')
#   plt.scatter(G750[:, i], RL750[:, i])  
#   plt.plot([0, 800], [0, 800], "r")

#   i = 2
#   grl50 = fig.add_subplot(349)  
#   grl50.set_xlabel('Obj val for Greedy (n = 50)')
#   grl50.set_ylabel('Obj val for RL (n = 50)', fontweight='bold')
#   plt.scatter(G50[:, i], RL50[:, i])
#   plt.plot([0, 45], [0, 45], "r")
#   grl200 = fig.add_subplot(3,4,10)
#   grl200.set_xlabel('Obj val for Greedy (n = 200)')
#   grl200.set_ylabel('Obj val for RL (n = 200)', fontweight='bold')
#   plt.scatter(G200[:, i], RL200[:, i])  
#   plt.plot([0, 130], [0, 130], "r")
#   grl500 = fig.add_subplot(3,4,11)
#   grl500.set_xlabel('Obj val for Greedy (n = 500)')
#   grl500.set_ylabel('Obj val for RL (n = 500)', fontweight='bold')
#   plt.scatter(G500[:, i], RL500[:, i])  
#   plt.plot([0, 300], [0, 300], "r")
#   grl750 = fig.add_subplot(3,4,12)
#   grl750.set_xlabel('Obj valfor Greedy (n = 750)')
#   grl750.set_ylabel('Obj val for RL (n = 750)', fontweight='bold')
#   plt.scatter(G750[:, i], RL750[:, i])  
#   plt.plot([0, 500], [0, 500], "r")
#   fig.tight_layout() 
#   plt.show()

