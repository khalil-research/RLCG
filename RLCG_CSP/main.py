import matplotlib
matplotlib.use('Agg')
import os.path
import numpy as np
from read_data import *
import Parameters 
from DQN import *
from test import *
from read_data import *
import random



seed_ = Parameters.seed
np.random.seed(seed_)
random.seed(seed_)



epsilon_ = Parameters.epsilon
decaying_epsilon_ = Parameters.decaying_epsilon
gamma_ = Parameters.gamma
alpha_ = Parameters.alpha_obj_weight
max_episode_num_ = Parameters.max_episode_num
min_epsilon_ = Parameters.min_epsilon
min_epsilon_ratio_ = Parameters.min_epsilon_ratio
capacity_ =  Parameters.capacity
hidden_dim_ = Parameters.hidden_dim
batch_size_ = Parameters.batch_size
epochs_ = Parameters.epochs
embedding_size_ = Parameters.embedding_size
cons_num_features_ = Parameters.cons_num_features
vars_num_features_ = Parameters.vars_num_features
learning_rate_ = Parameters.lr

display_ = True

model_index_ = Parameters.model_index

#### training and saving the data for plotting and model weights (weights and data are saved inside .learning)
schedule_train_name = "Name_files/scheduled_train.txt"

DQN = DQNAgent(env = None, capacity = capacity_, hidden_dim = hidden_dim_, batch_size = batch_size_, epochs = epochs_, embedding_size = embedding_size_, 
			   cons_num_features = cons_num_features_, vars_num_features = vars_num_features_, learning_rate = learning_rate_)




TRAIN = True
if TRAIN:
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)     
	total_times, episode_rewards, num_episodes =  DQN.learning(schedule_train_name, epsilon = epsilon_, decaying_epsilon = decaying_epsilon_, gamma = gamma_, 
	                learning_rate = learning_rate_, max_episode_num = max_episode_num_, display = display_, min_epsilon = min_epsilon_, min_epsilon_ratio = min_epsilon_ratio_,model_index = model_index_)    



TEST = False
if TEST:

	### here, what only matters is the parameters weight
	DQN_test = DQNAgent(env = None, capacity = 20000, hidden_dim = 32, batch_size = 32, epochs = 5, embedding_size = 32, 
				   cons_num_features = 2, vars_num_features = 9, learning_rate = 1e-3)

	DATA = general_compare(DQN_test,0,50)



## https://vrpsolver.math.u-bordeaux.fr/