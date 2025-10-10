from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm
import math
from typing import  Tuple
from Mamba_SSM import MambaSSM
import time 
import copy
import logging

torch.autograd.set_detect_anomaly(True)

from dsl import ALL_ACTIONS ,SHIFT_ACTIONS,TRANSFORM_ACTIONS
from helper import find_objects , extract_target_region
from helper_arc import  loader , display , get_module_logger
from helper_env import  placement , place_object
from helper_env import matrix_similarity
from dl_models.mamba import MambaBlock ,ModelArgs
from helper_arc import get_module_logger , plot_metrics
from A2C import A2CAgent




logger = get_module_logger(__name__)



# Initialized parameters
#----------------------------------


action_names = list(ALL_ACTIONS.keys())
shift_actions = list(SHIFT_ACTIONS.keys())
transform_actions=list(TRANSFORM_ACTIONS.keys())
num_actions = len(action_names)
OUTPUT = {}


# Initialized Clas
#--------------------------------------



class Example_Chooser:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.n = np.zeros(num_arms)  # Number of pulls per arm
        self.sum_r = np.zeros(num_arms)  # Sum of rewards per arm
        self.sum_sq = np.zeros(num_arms)  # Sum of squared rewards per arm
        self.solved_status = np.zeros(num_arms, dtype=bool)
    def select_example(self):

        if np.all(self.solved_status): return -1
        
        scores = np.full(self.num_arms, -np.inf)

        for i in range(self.num_arms):
            if self.solved_status[i]:
                continue

            if self.n[i] == 0:
                scores[i] = np.random.normal(0.0, 1.0) # Prior with mean 0, std 1
            else:
                mean = self.sum_r[i] / self.n[i]
                if self.n[i] == 1:
                    std = 1.0  # Default standard deviation for single sample
                else:
                    # Calculate sample variance and standard error
                    variance = (self.sum_sq[i] - self.n[i] * mean**2) / (self.n[i] - 1)
                    variance = max(variance, 1e-9) #to avoid sqrt of negative number due to float
                    std = np.sqrt(variance) / np.sqrt(self.n[i])
                
                scores[i] = np.random.normal(mean, std)
                logger.debug(f"Arm {i}: Mean={mean:.4f}, Std={std:.4f} -> Score={scores[i]:.4f}")           
        return int(np.argmax(scores))

    def update_arm(self, arm, reward):
        self.n[arm] += 1
        self.sum_r[arm] += reward
        self.sum_sq[arm] += reward**2

    def mark_as_solved(self, arm):
        print(f"Bandit: Marking example {arm} as solved.")
        self.solved_status[arm] = True


#Initialized functions
#---------------------------------------------------------------------


def Arc_Prize_Solver(examples,output,agent,max_iterations=100, max_steps_per_episode=4,min_iterations=50,patience=20):



    Placer = None #DQN_Solver(ft,len(examples[0]['output']),3)


    
    logger.debug(f"output shape & no of examples{len(examples[0]['output']),len(examples)}")

    num_examples = len(examples)

    bandit = Example_Chooser(num_examples)

    objects = None
    obj_list={}
    count =0

    best_score = -float('inf')
    
    steps_without_improvement = 0
    previous_shape=0
    iterations=0

    while iterations <= max_iterations:
        count += 1
        idx = bandit.select_example()
        if idx == -1 :   return example , True
        
        logger.debug(f'count{idx,count}')
        example = examples[idx]

        input_grid = np.array(example['input'])
        target_grid = np.array(example['output'])

        if input_grid.shape != previous_shape and previous_shape != 0 :
            logger.debug(f'Skipping example {idx} due to shape mismatch: {input_grid.shape} != {previous_shape}')
            continue

        previous_shape = input_grid.shape  # Update for next iteration
        logger.debug(f'input grid: {input_grid}')
        logger.debug(f'target grid: {target_grid}')
        solved = 0
        if idx not in obj_list:
            output[idx] = []
            predicted_grid = np.zeros_like(target_grid)
            example['predicted_grid'] = predicted_grid 
        else:
            predicted_grid = example['predicted_grid']

        logger.debug(f"'predicted_grid',{predicted_grid},{type(predicted_grid)}")
    
        old_reward = 0
        sim_score = 0
        

        # episode_best_score = -float('inf')
        # episode_steps_without_improvement = 0
        
        for step in range(max_steps_per_episode):
            new_grid, new_reward = find_solution(predicted_grid, agent , Placer, target_grid, objects)
            
            sim_score += new_reward - old_reward
            old_reward = new_reward
            
            if np.array_equal(new_grid, target_grid):
                solved += 1 
                bandit.mark_as_solved(idx)
                logger.debug(f'{idx} win no {solved} :{predicted_grid}')
                done=True
                break

            example['predicted_grid'] = new_grid
            output[idx].append((predicted_grid.tolist(), sim_score))
            print(sim_score)
                    
            agent.store_reward(new_reward, done)
                    

                    

            # Perform the update at the end of the episode
            agent.update()

            # --- Logging ---

            bandit.update_arm(idx, sim_score)

        
        # Update global early stopping tracking
        if sim_score > best_score:
            best_score = sim_score
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            
        iterations +=1
        if iterations >= min_iterations and steps_without_improvement >= patience * 2  :
 
            logger.info("warning -  no improvement across examples")
            return example, False


        elif iterations >= max_iterations:
            logger.info("No solution found within iterations")

            return example, False



#----------------------------------------------------------------------------------------------------



def find_solution(old_predicted_grid, agent , Placer_, target_grid,objects):

    agent.memory['states'].append([old_predicted_grid,obj_info['grid'],target_grid])

    episode_reward = 0

    action_idx = agent.select_action([old_predicted_grid,obj_info['grid'],target_grid])

    pos_values = [int(x * target_grid.shape[1]), int(y * target_grid.shape[0])]

            


        

    logger.debug(f'target_shape: {target_grid.shape} , new pos_values: {pos_values}')

    func = action_names[action_idx]
    is_place_action = False
    new_obj_info =obj_info.copy()

    if func == 'place' or obj_info['placed'] == False:


        new_obj_info['placed']=True
        is_place_action = True
        
        new_obj_info['position'] = pos_values

        objects.append(new_obj_info)
        obj_info = new_obj_info
    
    elif func in transform_actions:
        
        new_obj_info['grid']=ALL_ACTIONS[func](obj_info['grid'])

    elif func in shift_actions:
        new_obj_info['position']=ALL_ACTIONS[func](obj_info['position']) 

    new_predicted_grid =placement(old_predicted_grid, obj_info, new_obj_info, background=0)

    reward = matrix_similarity(new_obj_info['grid'],extract_target_region(target_grid,new_obj_info))

    if new_predicted_grid is None:
        new_predicted_grid = old_predicted_grid
        reward = 0


    h, w = target_grid.shape[:2]
    norm_pos = (
        new_obj_info['position'][0] / w,
        new_obj_info['position'][1] / h
    )
    logger.debug(f'new obj position{norm_pos}')



    obj_grid = place_object(np.zeros_like(target_grid.copy()),obj_info['grid'],obj_info['position'])
    new_obj_grid = place_object(np.zeros_like(target_grid.copy()),new_obj_info['grid'],new_obj_info['position'])



 
    reward= matrix_similarity(new_predicted_grid,target_grid)
    likelihood_predictor.store_experience(prob1,reward)

    return new_predicted_grid ,reward


if __name__ == "__main__":
    train, ids = loader(train_path='arc-prize-2025/arc-agi_training_challenges.json')
    count=0
    winning=0


    max_episodes = 2000
    max_timesteps = 500
    lr = 0.002
    gamma = 0.99
    entropy_beta = 0.01
    log_interval = 10
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"


    mamba_ssm = MambaSSM().to(device)
    agent = A2CAgent(mamba_ssm, len(action_names), lr, gamma, entropy_beta, device)

    for case_id in ids:


        count +=1
        start_time=time.time()
        # if count ==3:
        #     break

        
        task = train[case_id]
        examples = task['train'] 
  
        print(f"Processing task {case_id} with {len(examples)} examples")

        OUTPUT[case_id]={}
        

        # agent.load()
        

        example,success = Arc_Prize_Solver(examples,OUTPUT[case_id],agent=agent, max_iterations=25 , max_steps_per_episode=4,min_iterations=10)
                
        # agent.save()
        # agent.memory.clear()
        logger.debug(f'count: {count} time: {time.time()-start_time}')
        display(example['input'],example['output'],example['predicted_grid'])
        if success:
            print(f"Task {case_id} solved")
            
            logger.debug(f'won: {winning} ')
            winning +=1
        else:
            print(f"Task {case_id} not solved")


    with open('output.json', 'w') as f:
            json.dump(OUTPUT, f, indent=2)  
    print('no of winnings: ', winning)
    print('accruacy: ', winning/1000)