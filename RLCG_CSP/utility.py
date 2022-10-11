import numpy as np
from copy import deepcopy
from collections import namedtuple
from typing import List
from tqdm import tqdm
import random 


#### define transition object to hold the transition data
class Transition(object):
# s0_augmented, a0, r, is_done, s1_augmented, total

    def __init__(self, s0_augmented, a0, r:float, is_done:bool, s1_augmented,action_info_0,total_0, total_1):
          self.data = (s0_augmented, a0, r, is_done, s1_augmented, action_info_0, total_0, total_1)

    @property
    def s0(self): 
          return self.data[0]

    @property
    def a0(self): 
        return self.data[1]

    @property
    def reward(self): 
        return self.data[2]
    
    @property
    def is_done(self): 
        return self.data[3]

    @property
    def s1(self): 
        return self.data[4]
    
    @property
    def action_info_0(self): 
        return self.data[5]



    @property
    def total_0(self): 
        return self.data[6]

    @property
    def total_1(self): 
        return self.data[7]

    def __iter__(self):
        return iter(self.data)
    
    def __str__(self):
        edge_num1 = len(self.data[0][1][1])
        edge_num2 = len(self.data[-1][1][1])
        result = 'transit from bipartite graph with edge' + str(edge_num1) + ' to ' + str(edge_num2) +' collecting the reward ' + str(self.data[2])
            # return 'transit from bipartite graph ' + col_string1 + ' connects to ' + con_string1 + ' to ' + col_string2 + ' connects to ' + con_string2 +' collecting the reward ' + str(self.data[2])
        return result
        

    
############
def show_final_solution(cut):
    use = [math.ceil(i) for i in cut.ColumnSol_Val]
    for i, p in enumerate(cut.current_patterns):
        if use[i]>0:
            print('Pattern ', i, ': how often we should cut: ', use[i])
            print('----------------------')
            for j,order in enumerate(p):
                if order >0:
                    print('order ', j, ' how much: ', order)
            print()

    print('Total number of rolls used: ', int(np.asarray(cut.ColumnSol_Val).sum()))
##############


class Episode(object):
  def __init__(self, e_id:int = 0) -> None:
      self.total_reward = 0  
      self.trans_list = []    
      self.name = str(e_id)   

  def push(self, trans:Transition) -> float:
      self.trans_list.append(trans)
      self.total_reward += trans.reward 
      return self.total_reward

  @property
  def len(self):
      return len(self.trans_list)

  def __str__(self):
      return "episode {0:<4} {1:>4} steps,total reward:{2:<8.2f}".\
          format(self.name, self.len, self.total_reward)

  def print_detail(self):
      print("detail of ({0}):".format(self))
      for i,trans in enumerate(self.trans_list):
          print("step{0:<4} ".format(i),end=" ")
          print(trans)

  def pop(self) -> Transition:
      '''normally this method shouldn't be invoked.
      '''
      if self.len > 1:
          trans = self.trans_list.pop()
          self.total_reward -= trans.reward
          return trans
      else:
          return None

  def is_complete(self) -> bool:
      '''check if an episode is an complete episode
      '''
      if self.len == 0: 
          return False 
      return self.trans_list[self.len-1].is_done

  def sample(self, batch_size = 1):   
      '''随即产生一个trans
      '''
      return random.sample(self.trans_list, k = batch_size)

  def __len__(self) -> int:
      return self.len


class Experience(object):
  '''this class is used to record the whole experience of an agent organized
  by an episode list. agent can randomly sample transitions or episodes from
  its experience.
  '''
  def __init__(self, capacity:int = 20000):
      self.capacity = capacity    # 
      self.episodes = []          # 
      self.next_id = 0            # 
      self.total_trans = 0        # 
      
  def __str__(self):
      return "exp info:{0:5} episodes, memory usage {1}/{2}".\
              format(self.len, self.total_trans, self.capacity)

  def __len__(self):
      return self.len

  @property
  def len(self):
      return len(self.episodes)

  def _remove(self, index = 0):      
      '''
          remove an episode, defautly the first one.
          args: 
              the index of the episode to remove
          return:
              if exists return the episode else return None
      '''
      if index > self.len - 1:
          raise(Exception("invalid index"))
      if self.len > 0:
          episode = self.episodes[index]
          self.episodes.remove(episode)
          self.total_trans -= episode.len
          return episode
      else:
          return None

  def _remove_first(self):
      self._remove(index = 0)

  def push(self, trans): 

      if self.capacity <= 0:
          return
      while self.total_trans >= self.capacity: 
          episode = self._remove_first()
      cur_episode = None
      if self.len == 0 or self.episodes[self.len-1].is_complete():
          cur_episode = Episode(self.next_id)
          self.next_id += 1
          self.episodes.append(cur_episode)
      else:
          cur_episode = self.episodes[self.len-1]
      self.total_trans += 1
      return cur_episode.push(trans)      #return  total reward of an episode

  def sample(self, batch_size=32): # sample transition
      '''randomly sample some transitions from agent's experience.abs
      Transition
      args:
          number of transitions need to be sampled
      return:
          list of Transition.
      '''
      # sample_trans = []
      sample_trans = np.asarray([])
      for _ in range(batch_size):
          index = int(random.random() * self.len)
          sample_trans = np.append(sample_trans,self.episodes[index].sample())

          # sample_trans += self.episodes[index].sample()
      return sample_trans

  def sample_episode(self, episode_num = 1):  # sample episode
      '''
      '''
      return random.sample(self.episodes, k = episode_num)

  @property
  def last_episode(self):
      if self.len > 0:
          return self.episodes[self.len-1]
      return None

  

  
          


