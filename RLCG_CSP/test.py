## where the functions used for testing are stored; 
## these functions are used for generating the data needed in  plotting_functions.py testing functions

import numpy as np
import matplotlib.pyplot as plt
from env import * 
from DQN import *
from read_data import *
from gurobipy import GRB
import gurobipy as gp
import time




def follow_policy(DQN,action_info, s):
    '''DQN selects an action 
    '''
    total_added, Actions = action_info
    Q_s = DQN.target_Q(s)
    Q_s_for_action = Q_s[-total_added::]
    # rand_value = np.random.random()
    idx = int(np.argmax(Q_s_for_action))
    return Actions[idx]



def follow_expert(cut,patterns,action_num):
        
    n_pattern = len(patterns)
    pattern_range = range(n_pattern)
    order_range = range(cut.n)
    action_range = range(action_num)
    # patterns = np.asarray(patterns, dtype=int)
    master_problem = gp.Model("master problem")
    
    # decision variables
    lambda_ = master_problem.addVars(pattern_range,
                                    vtype=GRB.CONTINUOUS,
                                    obj=np.ones(n_pattern),
                                    name="lambda")
    
    y_ = master_problem.addVars(action_range,
                                vtype=GRB.BINARY,
                                obj=np.zeros(action_num),
                                name="yp")
    
    # direction of optimization (min or max)
    master_problem.modelSense = GRB.MINIMIZE
    

    # demand satisfaction constraint
    for i in order_range:

        master_problem.addConstr(sum(patterns[p][i]*lambda_[p] for p in pattern_range) == cut.demands[i],
                                 "Demand[%d]"%i)
    # total_n = n_pattern + action_num
    # print(total_n)

    for j in action_range:
        master_problem.addConstr(y_[j]>=lambda_[n_pattern-action_num+j],
                        "Indicator[%d]" %j)
        # print(n_pattern-action_num+j)

    master_problem.addConstr(sum(y_[k] for k in action_range) == 1,
                        "Select_one[%d]")  
     
    master_problem.Params.LogToConsole = 0

    master_problem.optimize()
    solution = master_problem.getAttr(GRB.Attr.X, master_problem.getVars())
    action_index = np.nonzero(solution[-action_num:])[0][0]
    return action_index


# ### models are saved as 
#       self.target_Q.save_state(path_model)
#       np.save(path_data+"model_"+str(model_index)+"_total_steps",np.asarray(total_times))
#       model_save_name = 'Model_'+str(model_index)+'.pt'
#       path_model = F'save_models/{model_save_name}'



# schedule_train_name = "Name_files/scheduled_train.txt"


def validation(DQN,model_index_):
    name_path = "Name_files/validation.txt"
    model_path = "save_models/"
    names_ = instance_name(name_path)
    total_results = []

    model_save_name = 'Model_'+str(model_index_)+'.pt'
    path_model = F'save_models/{model_save_name}'
    DQN.target_Q.restore_state(path_model)
    DQN.behavior_Q.restore_state(path_model)

    each_model_result = []
    for k in range(len(names_)):
        cut2 = instance_val(i,name_file)
        reward, is_done = cut2.initialize()
        time = time
        DQN.env = cut2
        DQN.S = DQN.get_aug_state()
        
        while True:
            if is_done:
                break

            action_info = DQN.S[1]
            s = DQN.S[0]
            action = follow_policy(DQN,action_info,s)
            reward, is_done = cut2.step(action,False)
            DQN.S = DQN.get_aug_state()

        history_opt_rl = cut2.objVal_history
        time3 = time.time() 
        obj_RL = history_opt_rl[-1]
        each_model_result.append(len(history_opt_rl))

    # total_results.append(each_model_result)

    np.save("save_data/validation_data/model_"+str(model_index_)+"_val",each_model_result)




def general_compare(DQN_test,model_index,prob_size):

    prob_size = int(prob_size)
    name = "TestName"+str(prob_size)
    name_file = "Name_files/"+name+".txt"

    names = instance_name(name_file)
    # print("names",names)
    total_length = len(names)

    Greedy=[]
    Expert = []
    RL=[]

    True_obj = [] ## do we put it here and have this for all instances ? 

    print("#####################")
    print("Starts testing for model "+str(model_index)+" for instance size "+str(prob_size))
    print()

    for i in range(total_length):
        # try:

            ## used for reading optimal values from excel files
            # name_of_instance = names[i]
            # opt_value_path = ''
        try:
            ################################## Greedy
            time1 = time.time() 
            cut1 = instance_test(i,name_file)
            reward, is_done = cut1.initialize()
            # print("starts greedy")
            while True:
                if is_done:
                    break

                action = cut1.available_action[0][0]
                reward, is_done = cut1.step(action,False)

            history_opt_g = cut1.objVal_history
            time2 = time.time() 
            obj_greedy = history_opt_g[-1]
            steps_g = len(history_opt_g)
            print("Greedy takes {} steps to reach obj {} with time {}".format(steps_g,obj_greedy,time2-time1))
            # print("starts RL")
            
            ###################################  RL
            cut2 = instance_test(i,name_file)
            reward, is_done = cut2.initialize()

     
            model_save_name = 'Model_'+str(model_index)+'.pt'
            path_model = F'save_models/{model_save_name}'
            DQN_test.target_Q.restore_state(path_model)
            DQN_test.behavior_Q.restore_state(path_model)

            DQN_test.env = cut2
            DQN_test.S = DQN_test.get_aug_state()

            while True:
                if is_done:
                    break


                action_info = DQN_test.S[1]
                s = DQN_test.S[0]
                action = follow_policy(DQN_test,action_info,s)
                reward, is_done = cut2.step(action,False)
                DQN_test.S = DQN_test.get_aug_state()

            history_opt_rl = cut2.objVal_history
            time3 = time.time() 
            obj_RL = history_opt_rl[-1]
            steps_RL = len(history_opt_rl)
            print("RL takes {} steps to reach obj {} with time {}".format(steps_RL,obj_RL,time3-time2))
            
            ###### Compare with expert
            cut3 = instance_test(i,name_file)
            reward, is_done = cut3.initialize()

            while True:
                if is_done:
                    break

                actions = cut3.available_action[0]
                patterns = deepcopy(cut3.current_patterns)
        
                for act in actions:
                    patterns.append(act)
                # print(patterns)
                action_index = follow_expert(cut3,patterns,len(actions))
                real_act = cut3.available_action[0][action_index]
                reward, is_done = cut3.step(real_act)

            history_opt_expert = cut3.objVal_history
            obj_expert = history_opt_expert[-1]
            steps_expert = len(history_opt_expert)
            time4 = time.time() 
            # except:
            #     continue

            print("Expert takes {} steps to reach obj {} with time {}".format(steps_expert,obj_expert,time4-time3))

        except:
            continue

        #### (full history, total number of steps, times, obj value)
        Greedy.append((history_opt_g,len(history_opt_g),time2-time1,obj_greedy))
        RL.append((history_opt_rl,len(history_opt_rl),time3-time2,obj_RL))
        Expert.append((history_opt_expert,len(history_opt_expert),time4-time3,obj_expert))
        print()
        print("{} steps out of {}".format(i,total_length))
        print("#########")


    complete_data = (Greedy,RL,Expert)
    path = 'save_data/testing_data/'
    np.save(path+'testing_result_model_'+str(model_index)+"_size"+str(prob_size),complete_data)

    #### return three lists of tuple: (running time, steps)

    return complete_data


