import Parameters
import numpy as np
from agents import Agent
from net import BipartiteGNN
from copy import deepcopy
import random


class DQNAgent(Agent):
    '''
    '''
    def __init__(self, env,
                       capacity,
                       hidden_dim,
                       batch_size,
                       epochs,
                       embedding_size,
                       cons_num_features,
                       vars_num_features,
                       learning_rate):

        super(DQNAgent, self).__init__(env, capacity)
        self.embedding_size = embedding_size 
        # self.hidden_dim = hidden_dim
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.lr = learning_rate

        self.behavior_Q = BipartiteGNN(embedding_size = self.embedding_size, cons_num_features = self.cons_num_features, 
        vars_num_features = self.vars_num_features, learning_rate = self.lr)

        self.target_Q = BipartiteGNN(embedding_size = self.embedding_size, cons_num_features = self.cons_num_features, 
        vars_num_features = self.vars_num_features, learning_rate = self.lr)
        self._update_target_Q()
        
        self.batch_size = batch_size 
        self.epochs = epochs



        
    def _update_target_Q(self):
       
        '''
        '''
        self.target_Q.set_weights(deepcopy(self.behavior_Q.variables))
        # self.target_Q = self.behavior_Q.clone()
        
    
    ## s is the super s0, A is the list containing all actions
    def policy(self, action_info, s, epsilon = None):

        total_added, Actions = action_info
        Q_s = self.behavior_Q(s)
        Q_s_for_action = Q_s[-total_added::]

        rand_value = np.random.random()
        if epsilon is not None and rand_value < epsilon:
            return random.choice(list(Actions))
        else:
            idx = int(np.argmax(Q_s_for_action))
            return Actions[idx]


    ## s is the super s0, A is the list containing all actions
    ### need action info 0 and total 1 (total_1 to get max Q_1, action_info_0 to get update index)
    def get_max(self,  total_1, s):
        Q_s = self.target_Q.call(s)
        Q_s_for_action = Q_s[-total_1::]
        return np.max(Q_s_for_action)

    ## this method is used to get target in _learn_from_memory function
    ## select the max number from last few items from Q_matrix for each row, based on last_index_list
    
    def _learn_from_memory(self, gamma, learning_rate):

        ## trans_pieces is a list of transitions
        trans_pieces = self.sample(self.batch_size)  # Get transition data
        states_0 = np.vstack([x.s0 for x in trans_pieces]) # as s0 is a list, so vstack
        actions_0 = np.array([x.a0 for x in trans_pieces]) 
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_dones = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])
        action_info = np.vstack([x.action_info_0 for x in trans_pieces])
        totals_0 = np.vstack([x.total_0 for x in trans_pieces])
        totals_1 = np.vstack([x.total_1 for x in trans_pieces])

        y_batch = []
        for i in range(len(states_0)):
          
            ### get the index of action that is taken at s0
            acts_0 = action_info[i][1]
            act_0 = list(actions_0[i])

            idx = 0
            for act in acts_0:
                if (act==act_0).all():
                    break
                idx+=1

            y = self.target_Q.call(states_0[i]).numpy()
            #### set the non action terms to be 0 
            y[0:-totals_0[i][0]] = 0

            if is_dones[i]:
                Q_target = reward_1[i]
            else:
                ### the number of actions for state 1 is used to get Q_target
                Q_max = self.get_max(totals_1[i][0], states_1[i])
                Q_target = reward_1[i] + gamma * Q_max

            y[-totals_0[i][0]+idx] = Q_target
            
            y_batch.append(np.asarray(y))

        y_batch= np.asarray(y_batch)
        X_batch = states_0
        
        loss = self.behavior_Q.train_or_test(X_batch, y_batch, totals_0, actions_0, action_info, True)
        # print("The loss is,", loss)
        self._update_target_Q()

        return loss

    #### the learning code
    def learning_method(self, instance, gamma, learning_rate, epsilon, 
                        display):

        epochs = self.epochs
        ###########
        self.env = instance
        self.S = self.get_aug_state()
        
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            s0_aug = self.S[0]
            action_info = self.S[1]
            ### a0 is selected based on behavior_Q
            # print(action_info[1])
            if len(action_info[1]) == 0:
                ## if no available actions,end this episode
                break;
            
            a0 = self.policy(action_info, s0_aug, epsilon)

            s1_augmented, r, is_done, total_reward = self.act(a0)


            ############################################################################################################
            if self.total_trans > self.batch_size:
                for e in range(epochs):
                    loss += self._learn_from_memory(gamma, learning_rate)
            ############################################################################################################
            
                # loss/=epochs
            # s0 = s1
            time_in_episode += 1

        loss /= (time_in_episode*epochs)
        if display:
            print("epsilon:{:3.2f},loss:{:3.2f},{}".format(epsilon,loss,self.experience.last_episode))
        return time_in_episode, total_reward  
