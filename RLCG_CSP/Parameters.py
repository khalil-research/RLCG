import numpy as np

## seed
seed = 0


## parameters about neural network
lr = 3e-4 ## https://www.jeremyjordan.me/nn-learning-rate/ (best lr)
batch_size = 32
capacity = 20000
hidden_dim = 32
epochs = 5 ## 
embedding_size = 32
cons_num_features = 2
vars_num_features = 9



## parameters of RL algorithm
gamma = 0.999
epsilon = 0.01
min_epsilon = 1e-2
min_epsilon_ratio = 0.8
decaying_epsilon = False
step_penalty = 1
alpha_obj_weight = 100
action_pool_size = 10 ## solution pool
max_episode_num = 439 ## as there are 440 training instances## as there are 440 training instances
capacity = 2000 ## so that the experience is relatively new




## parameter index
model_index = 0



