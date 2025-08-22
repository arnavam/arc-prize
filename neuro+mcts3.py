from RL_alg.BaseDQN import BaseDQN
import numpy as np
import time 
import os
from dsl import find_objects ,extract_target_region, COMB ,ACTIONS,PRIMITIVE
from A_arc import  loader , display
from RL_alg.reinforce import FeatureExtractor
from RL_alg.DQNAction_Classifier import DQN_Solver_MultiHead
from RL_alg.ReinLikelihood import Likelihood

from env import  placement
from env import matrix_similarity
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to current_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)

# Initialized parameters
#----------------------------------
action_names = list(COMB.keys())
names1 = list(ACTIONS.keys())
names2=list(PRIMITIVE.keys())
num_actions = len(action_names)

ft = FeatureExtractor(input_channels=1)
neuro_classifer = DQN_Solver_MultiHead(ft, num_actions,3)
Likeliehood = Likelihood(ft, 1,3)

# Initialized Clas
#--------------------------------------
class GaussianMultiArmBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.n = np.zeros(num_arms)  # Number of pulls per arm
        self.sum_r = np.zeros(num_arms)  # Sum of rewards per arm
        self.sum_sq = np.zeros(num_arms)  # Sum of squared rewards per arm

    def select_arm(self):
        scores = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            if self.n[i] == 0:
                scores[i] = np.random.normal(0, 1)  # Prior with mean 0, std 1
            else:
                mean = self.sum_r[i] / self.n[i]
                if self.n[i] == 1:
                    std = 1.0  # Default standard deviation for single sample
                else:
                    # Calculate sample variance and standard error
                    variance = (self.sum_sq[i] - self.n[i] * mean**2) / (self.n[i] - 1)
                    std = np.sqrt(variance) / np.sqrt(self.n[i])
                scores[i] = np.random.normal(mean, std)
        return np.argmax(scores)

    def update_arm(self, arm, reward):
        self.n[arm] += 1
        self.sum_r[arm] += reward
        self.sum_sq[arm] += reward**2


#Initialized functions
#---------------------------------------------------------------------

def find_solution(current_grid, Placer_, target_grid,objects):

 
    idx,prob1 = Likeliehood.select_action([current_grid, objects, target_grid])
    obj_info  = objects[idx]

    action_idx ,pos_values= neuro_classifer.select_action([current_grid,  obj_info['grid'], obj_info['position'],target_grid])

    pos_values = [int(pos_values[0] * target_grid.shape[1]), int(pos_values[1] * target_grid.shape[0])]
    print('target_shape',target_grid.shape)
    print('new pos_values',pos_values)
    func= action_names[action_idx]
    is_place_action = False
    if func == 'idle' or obj_info['placed'] == False:

        new_obj_info =obj_info.copy()
        obj_info['placed']=True
        is_place_action = True
        new_obj_info['position'] = pos_values

        objects.append(new_obj_info)
        obj_info = new_obj_info
    
    elif func in names2:
        
        new_obj_info =obj_info.copy()
        new_obj_info['grid']=COMB[func](obj_info['grid'])

    elif func in names1:
        new_obj_info = obj_info.copy()
        new_obj_info['position']=COMB[func](obj_info['position']) 

    new_grid =placement(current_grid, obj_info, new_obj_info, background=0)

    if new_grid is None:
        new_grid = current_grid

    reward = matrix_similarity(new_obj_info['grid'],extract_target_region(target_grid,new_obj_info))

    h, w = target_grid.shape[:2]
    norm_pos = (
        new_obj_info['position'][0] / w,
        new_obj_info['position'][1] / h
    )
    print(new_obj_info['position'])

    neuro_classifer.store_experience(
        state=(current_grid, obj_info['grid'], obj_info['position']),
        action=action_idx,
        reward=reward,
        next_state=(new_grid, new_obj_info['grid'], new_obj_info['position']),
        true_position=norm_pos,
        is_place_action=is_place_action
    )


    # Update policy periodically
    neuro_classifer.update_policy()

    reward= matrix_similarity(new_grid,target_grid)
    Likeliehood.store_experience(prob1,reward)

    return new_grid ,reward

def Gaussian_multi_armbandit(examples, max_iterations=1000, max_steps_per_episode=4):


    shape=len(examples[0]['output'])
    Placer = None#DQN_Solver(ft,shape,3)
    num_examples = len(examples)
    print(shape,num_examples)
    bandit = GaussianMultiArmBandit(num_examples)
    objects = None
    obj_list={}
    count =0
    for iteration in range(max_iterations):
        count+=1
        idx = bandit.select_arm()
        print('count',idx,count)
        example = examples[idx]
        input_grid = np.array(example['input'])
        target_grid = np.array(example['output'])

        print('input grid: ',input_grid)
        print('target grid: ',target_grid)

        if example.get('current_grid') is None:
            objects = find_objects(input_grid)
            obj_list[idx]=objects
            current_grid = np.zeros_like(target_grid)
            example['current_grid'] = current_grid 

        objects=obj_list[idx]
        current_grid=example['current_grid']
        print('current_grid',current_grid,type(current_grid))

        new_grid = current_grid
        old_reward=0
        sim_score=0
        for step in range(max_steps_per_episode):
            new_grid ,new_reward= find_solution(new_grid, Placer, target_grid,objects)
            
            sim_score += new_reward-old_reward
            old_reward=new_reward
                    

            if np.array_equal(new_grid, target_grid):
                #remove the current example from the iterations
                #add solved +=1 
                #if no more to remove say we solved everyone and use maybe testing example to test the solution
                break

        example['current_grid']= current_grid

        print(sim_score)
        neuro_classifer.update_policy()
        Likeliehood.update_policy()
        bandit.update_arm(idx, sim_score)
        
        # Check every 10 iterations if the current solution works for all examples
    #     if iteration % 10 == 0:
    #         all_solved = True
    #         for ex in examples:
    #             test_input = ex['input']
    #             test_target = ex['output']
    #             test_grid = test_input
    #             for step in range(max_steps_per_episode):
    #                 test_grid = find_solution(test_grid, Placer, test_target,objects)
    #                 if np.array_equal(test_grid, test_target):
    #                     break
    #             if not np.array_equal(test_grid, test_target):
    #                 all_solved = False
    #                 break
    #         if all_solved:
    #             logging.info(f"Solution found after {iteration} iterations")
    #             return True  # Solution found
    
    # logging.info("No solution found within iterations")
    return False  # No solution found

if __name__ == "__main__":
    train, ids = loader(train_path='arc-prize-2025/arc-agi_training_challenges.json')
    count=0
    for case_id in ids:
        count +=1
        if count ==2:
            break
        task = train[case_id]
        examples = task['train']  # Assume each task has a 'train' list of examples
        print(examples)
        logging.debug(f"Processing task {case_id} with {len(examples)} examples")
        success = Gaussian_multi_armbandit(examples)
        if success:
            print(f"Task {case_id} solved")
        else:
            print(f"Task {case_id} not solved")