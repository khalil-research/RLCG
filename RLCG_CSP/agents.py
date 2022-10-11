
import random
import numpy as np
from utility import *
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from read_data import *
from tqdm import tqdm
import matplotlib.pyplot as plt





class Agent(object):
  '''Base Class of Agent
  '''
  def __init__(self, initial_env=None, capacity = 10000):
      self.env = initial_env # the evironment would be one cutting stock object
      ## add the env and available action will be added in the learning_method 
      # self.A = self.env.available_action

      self.A = []
      self.experience = Experience(capacity = capacity)
      # S record the current super state for the agent
      # self.S = self.get_aug_state()   

      self.S = []


  ## get augmented state from the current environment
  def get_aug_state(self):


      actions,reduced_costs = deepcopy(self.env.available_action)
      total_added = len(actions)
      patterns = self.env.current_patterns[:]

      is_action = np.asarray([0]*len(patterns))

      patterns.extend(actions)
      Is_action = np.zeros(len(patterns))


      col_num = len(patterns)
      cons_num = self.env.n
      column_features = []
      cons_features = []
      edge_indices = [[],[]]


      #### column features, also augment all actions
      Is_action = np.append(Is_action,np.ones(total_added))
      RC = self.env.RC[:]
      RC = np.append(RC,reduced_costs)
      In_Cons_Num = self.env.In_Cons_Num[:]
      
      actions = np.asarray(actions)
      if actions.ndim == 1:
          In_Cons_Num = np.append(In_Cons_Num, np.count_nonzero(actions))
      else:
          In_Cons_Num = np.append(In_Cons_Num, np.count_nonzero(np.asarray(actions),axis=1))

      ColumnSol_Val = self.env.ColumnSol_Val[:]
      ColumnSol_Val = np.append(ColumnSol_Val, np.zeros(total_added))
      Waste = self.env.Waste[:]
      Waste = np.append(Waste, [self.env.roll_len-np.dot(np.asarray(actions[i]),np.asarray(self.env.order_lens)) for i in range(total_added)])
      stay_in = self.env.stay_in[:]
      stay_in = np.append(stay_in,np.zeros(total_added))
      stay_out = self.env.stay_out[:]
      stay_out = np.append(stay_out,np.zeros(total_added))
      just_left = self.env.just_left[:]
      just_left = np.append(just_left,np.zeros(total_added))
      just_enter = self.env.just_enter[:]
      just_enter = np.append(just_enter,np.zeros(total_added))
      is_action = np.append(is_action,np.ones(total_added))
      


      #### constraint features, also augment all actions
      Shadow_Price = self.env.Shadow_Price[:]
      In_Cols_Num = self.env.In_Cols_Num[:]
      for action in actions:
          non_zero = np.nonzero(action)
          for idx in non_zero:
              In_Cols_Num[idx]+=1
    
    
      RC = np.asarray(RC).reshape(-1, 1)
      Shadow_Price = np.asarray(Shadow_Price).reshape(-1, 1)
      In_Cons_Num = np.asarray(In_Cons_Num).reshape(-1, 1)
      In_Cols_Num = np.asarray(In_Cols_Num).reshape(-1, 1)
      ColumnSol_Val = np.asarray(ColumnSol_Val).reshape(-1, 1)
      Waste = np.asarray(Waste).reshape(-1, 1)
      stay_in = np.asarray(stay_in).reshape(-1, 1)
      stay_out = np.asarray(stay_out).reshape(-1, 1)


      from sklearn.preprocessing import MinMaxScaler
      # from sklearn.preprocessing import StandardScalar

      Scaler_RC = MinMaxScaler()
      Scaler_RC.fit(RC)
      RC = Scaler_RC.transform(RC)
      Scaler_SP = MinMaxScaler()
      Scaler_SP.fit(Shadow_Price)
      Shadow_Price = Scaler_RC.transform(Shadow_Price)
      Scaler_IConsN = MinMaxScaler()
      Scaler_IConsN.fit(In_Cons_Num)
      In_Cons_Num = Scaler_IConsN.transform(In_Cons_Num)
      Scaler_IColsN = MinMaxScaler()
      Scaler_IColsN.fit(In_Cols_Num)
      In_Cols_Num = Scaler_IColsN.transform(In_Cols_Num)
      Scaler_CSV = MinMaxScaler()
      Scaler_CSV.fit(ColumnSol_Val)
      ColumnSol_Val = Scaler_CSV.transform(ColumnSol_Val)
      Scaler_W = MinMaxScaler()
      Scaler_W.fit(Waste)
      Waste = Scaler_W.transform(Waste)

      Scaler_si = MinMaxScaler()
      Scaler_si.fit(stay_in)
      stay_in = Scaler_si.transform(stay_in)

      Scaler_out = MinMaxScaler()
      Scaler_out.fit(stay_out)
      stay_out = Scaler_out.transform(stay_out)

      RC = list(RC.T[0])
      Shadow_Price = list(Shadow_Price.T[0])
      In_Cons_Num = list(In_Cons_Num.T[0])
      In_Cols_Num = list(In_Cols_Num.T[0])
      ColumnSol_Val = list(ColumnSol_Val.T[0])
      Waste = list(Waste.T[0])
      stay_in = list(stay_in.T[0])
      stay_out = list(stay_out.T[0])


      ### constraint nodes
      for j in range(cons_num):
          con_feat = []
          con_feat.append(Shadow_Price[j])
          con_feat.append(In_Cols_Num[j])
          cons_features.append(con_feat)

      
      ### normalize here for each information
      for i in range(col_num):
          col_feat = []
          col_feat.append(RC[i])
          col_feat.append(In_Cons_Num[i])
          col_feat.append(ColumnSol_Val[i])
          col_feat.append(Waste[i])
          col_feat.append(stay_in[i])
          col_feat.append(stay_out[i])
          col_feat.append(just_left[i])
          col_feat.append(just_enter[i])
          col_feat.append(is_action[i])

          column_features.append(col_feat)
      
      
      for m in range(len(patterns)):
          for n in range(len(patterns[0])):
              if patterns[m][n]!=0:
                  # then mth column is connected to nth cons
                  edge_indices[0].append(m)
                  edge_indices[1].append(n)

      edge_indices = np.asarray(edge_indices)
      edge_indices[[0, 1]] = edge_indices[[1, 0]]
      cons_features=np.asarray(cons_features)
      column_features=np.asarray(column_features)

      ## need this total_added for reading the Q values, need actions to select onne pattern after read Q values
      aug_state, action_info = ((cons_features, edge_indices, column_features),(total_added,actions))
      return aug_state, action_info

  def policy(self):

      return random.choice(self.A)
      # return random.sample(self.A, k=1)[0]
  
  def perform_policy(self, s, Q = None, epsilon = 0.05):
      action = self.policy()
      return action


  def act(self, a0):
      ## get the current super state 
      s0_augmented, action_info_0 = self.S
      total_0 = deepcopy(action_info_0[0])
      # print(s0_augmented)

      ## step change the environnment, update all the information used for agent to construct state
      r, is_done = self.env.step(a0)

      s1_augmented, action_info_1 = self.get_aug_state()
      total_1 = action_info_1[0]
      trans = Transition(s0_augmented, a0, r, is_done, s1_augmented, action_info_0, total_0, total_1)
      total_reward = self.experience.push(trans)
      self.S = s1_augmented, action_info_1

      return s1_augmented, r, is_done, total_reward

  def learning_method(self,cut_stock_instance, gamma = 0.9, alpha = 1e-3, epsilon = 0.05):
      #self.state = self.env.reset()
      ## initialize an environment
      self.env = cut_stock_instance

      ### initialize before calling
      # self.env.initialize()

      self.A = self.env.available_action[0]

      s0 = self.S
      a0 = self.perform_policy(s0, epsilon)
      time_in_episode, total_reward = 0, 0
      is_done = False
      while not is_done:
          # act also update self.S
          s1, r1, is_done, total_reward = self.act(a0)
          self.A = self.env.available_action[0]
          # if self.A == []:
          #     break;
          a1 = self.perform_policy(s1, epsilon)
          s0, a0 = s1, a1
          time_in_episode += 1

          # actions,reduced_costs = deepcopy(self.env.available_action)

      return time_in_episode, total_reward  
    

  def _decayed_epsilon(self,cur_episode: int, 
                            min_epsilon: float, 
                            max_epsilon: float, 
                            target_episode: int) -> float: 
      slope = (min_epsilon - max_epsilon) / (target_episode)
      intercept = max_epsilon
      return max(min_epsilon, slope * cur_episode + intercept)        
      
                      
  def learning(self, name_file,  epsilon = 0.05, decaying_epsilon = True, gamma = 0.9, 
                learning_rate = 3e-4, max_episode_num = 439, display = False, min_epsilon = 1e-2, min_epsilon_ratio = 0.8, model_index = 0):
      total_time,  episode_reward, num_episode = 0,0,0
      total_times, episode_rewards, num_episodes = [], [], []

      # max_episode_num now set to be 480 as there are 480 instances uploaded

      # print("max_episode_num is",max_episode_num)
      for i in range(max_episode_num):
          if epsilon is None:
              epsilon = 1e-10
          elif decaying_epsilon:
              #epsilon = 1.0 / (1 + num_episode)
              epsilon = self._decayed_epsilon(cur_episode = num_episode+1,
                                              min_epsilon = min_epsilon,
                                              max_epsilon = 0.05,
                                              target_episode = int(max_episode_num * min_epsilon_ratio))
              
          #### read_file    
          cut_stock_instance = instance_train(i,name_file)
          # name_ = name_file[i]
          # optimal_val = df.loc[df['Name'] == "BPP_50_50_0.1_0.8_0.txt", "Best LB"].item()

          if cut_stock_instance == "not found":
              print("########### NOT FOUND ###############")
              continue;
    

          cut_stock_instance.initialize()
          
          time_in_episode, episode_reward = self.learning_method(cut_stock_instance,\
                gamma = gamma, learning_rate = learning_rate, epsilon = epsilon, display = display)
          # total_time += time_in_episode
          num_episode += 1

          total_times.append(time_in_episode)
          episode_rewards.append(episode_reward)
          num_episodes.append(num_episode)



          print("Episode: " + str(num_episode) + " takes " + str(time_in_episode) +" steps with epsilon "+str(epsilon))

          if num_episode%80==0:

            
              model_save_name = 'Model_'+str(model_index)+"_"+str(num_episode)+'.pt'
              path_model_check = F'check_points/Model/{model_save_name}'
              self.target_Q.save_state(path_model_check)


              path_data_check = 'check_points/Data/'
              np.save(path_data_check+"model_"+str(model_index)+"_total_steps_"+str(num_episode),np.asarray(total_times))


              fig, axs = plt.subplots(2)
              fig.suptitle('Vertically stacked subplots steps, reward')
              axs[0].plot(num_episodes, total_times)
              axs[1].plot(num_episodes, episode_rewards)
              plt.savefig("./save_graph/training_plots/"+str(num_episode)+".png")


      path_data = 'save_data/training_data/'
      model_save_name = 'Model_'+str(model_index)+'.pt'
      path_model = F'save_models/{model_save_name}'

      self.target_Q.save_state(path_model)
      np.save(path_data+"model_"+str(model_index)+"_total_steps",np.asarray(total_times))





      return  total_times, episode_rewards, num_episodes

  def sample(self, batch_size = 32):

      return self.experience.sample(batch_size)

  @property
  def total_trans(self):

      return self.experience.total_trans
  
  def last_episode_detail(self):
      self.experience.last_episode.print_detail()

