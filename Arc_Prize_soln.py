import numpy as np
import time 
import os
import json
import torch
import copy
import logging

torch.autograd.set_detect_anomaly(True)

from dsl import ALL_ACTIONS ,SHIFT_ACTIONS,TRANSFORM_ACTIONS
from helper import find_objects , extract_target_region
from helper_arc import  loader , display , get_module_logger
from helper_env import  placement , place_object
from helper_env import matrix_similarity
from dl_models.feature_extractor import FeatureExtractor
from dl_models.DQNAction_Classifier import DQN_Classifier
from dl_models.ReinLikelihood import Likelihood

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
                logging.debug(f"Arm {i}: Mean={mean:.4f}, Std={std:.4f} -> Score={scores[i]:.4f}")           
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


def Arc_Prize_Solver(examples,output,action_classifier,likelihood_predictor,max_iterations=100, max_steps_per_episode=4,):



    Placer = None #DQN_Solver(ft,len(examples[0]['output']),3)


    
    logging.debug(f"output shape & no of examples{len(examples[0]['output']),len(examples)}")

    num_examples = len(examples)

    bandit = Example_Chooser(num_examples)

    objects = None
    obj_list={}
    count =0
    # Add these parameters at the beginning of your function or as function parameters
    min_steps = 50  # Minimum number of steps to run
    patience = 20   # Number of steps to wait without improvement before stopping
    best_score = -float('inf')
    steps_without_improvement = 0

    for iteration in range(max_iterations):
        count += 1
        idx = bandit.select_example()
        if idx == -1:    return example['predicted_grid'], True
        
        logging.debug(f'count{idx,count}')
        example = examples[idx]

        input_grid = np.array(example['input'])
        target_grid = np.array(example['output'])

        logging.debug(f'input grid: {input_grid}')
        logging.debug(f'target grid: {target_grid}')
        solved = 0
        if idx not in obj_list:
            output[idx] = []
            objects = find_objects(input_grid)
            obj_list[idx] = objects
            predicted_grid = np.zeros_like(target_grid)
            example['predicted_grid'] = predicted_grid 
        else:
            objects = obj_list[idx]   
            predicted_grid = example['predicted_grid']

        logging.debug(f"'predicted_grid',{predicted_grid},{type(predicted_grid)}")
        
        old_reward = 0
        sim_score = 0
        
        # Early stopping initialization for this episode
        episode_best_score = -float('inf')
        episode_steps_without_improvement = 0
        
        for step in range(max_steps_per_episode):
            new_grid, new_reward = find_solution(predicted_grid, likelihood_predictor, action_classifier, Placer, target_grid, objects)
            
            sim_score += new_reward - old_reward
            old_reward = new_reward
            
            # Check for improvement
            if new_reward > episode_best_score:
                episode_best_score = new_reward
                episode_steps_without_improvement = 0
            else:
                episode_steps_without_improvement += 1
            
            # Early stopping condition (but only after minimum steps)
            if step >= min_steps and episode_steps_without_improvement >= patience:
                logging.debug(f"Early stopping at step {step} for example {idx}")
                break
                
            if np.array_equal(new_grid, target_grid):
                solved += 1 
                bandit.mark_as_solved(idx)
                logging.debug(f'{idx} win no {solved} :{predicted_grid}')
                break

            example['predicted_grid'] = new_grid
            output[idx].append((predicted_grid.tolist(), sim_score))
            print(sim_score)
            action_classifier.update_policy()
            likelihood_predictor.update_policy()
            bandit.update_arm(idx, sim_score)
        
        # Update global early stopping tracking
        if sim_score > best_score:
            best_score = sim_score
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            
        # Global early stopping (optional)
        if iteration >= min_steps and steps_without_improvement >= patience * 2:
            logging.info("warning -  no improvement across examples")

            return predicted_grid, False

    logging.info("No solution found within iterations")

    return predicted_grid, False

#----------------------------------------------------------------------------------------------------



def find_solution(old_predicted_grid, likelihood_predictor,action_classifier, Placer_, target_grid,objects):

 
    idx,prob1 = likelihood_predictor.select_action([old_predicted_grid, objects, target_grid])
    obj_info  = objects[idx]

    action_idx ,(x,y)= action_classifier.select_action([old_predicted_grid,obj_info['grid'],target_grid])

    pos_values = [int(x * target_grid.shape[1]), int(y * target_grid.shape[0])]

    logging.debug(f'target_shape: {target_grid.shape} , new pos_values: {pos_values}')

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
    logging.debug(f'new obj position{norm_pos}')



    obj_grid = place_object(np.zeros_like(target_grid.copy()),obj_info['grid'],obj_info['position'])
    new_obj_grid = place_object(np.zeros_like(target_grid.copy()),new_obj_info['grid'],new_obj_info['position'])
    action_classifier.store_experience(
        state=(old_predicted_grid, obj_grid),

        action=action_idx,
        reward=reward,
        next_state=(new_predicted_grid, new_obj_grid),
        true_position=norm_pos,
        is_place_action=is_place_action
    )


 
    reward= matrix_similarity(new_predicted_grid,target_grid)
    likelihood_predictor.store_experience(prob1,reward)

    return new_predicted_grid ,reward


if __name__ == "__main__":
    train, ids = loader(train_path='arc-prize-2025/arc-agi_training_challenges.json')
    count=0
    winning=0
    start_id = '009d5c81'

    # Find the index of the start_id
    start_index = ids.index(start_id)

    # Iterate from the specified start_id
    for case_id in ids[start_index:]:


        # count +=1
        # if count ==3:
        #     break

        task = train['00d62c1b']
        examples = task['train']  # Assume each task has a 'train' list of examples
        print(examples)
        for i in range(4):
            a = task['train'][i]['input']
            b = task['train'][i]['output']

            logger.debug(f'{np.matrix(a)}]\n {np.matrix(b)}')
        logging.debug(f"Processing task {case_id} with {len(examples)} examples")

        OUTPUT[case_id]={}
        
        ft1 = FeatureExtractor(input_channels=1)
        action_classifier = DQN_Classifier(feature_extractor=ft1, no_of_outputs=num_actions,no_of_inputs=3,device='cuda')

        ft2 = FeatureExtractor(input_channels=1)
        likelihood_predictor = Likelihood(feature_extractor= ft2, no_of_outputs=1,no_of_inputs=3,device='cuda')

        action_classifier.load()
        likelihood_predictor.load()

        predicted , success = Arc_Prize_Solver(examples,OUTPUT[case_id], action_classifier,likelihood_predictor ,max_iterations=100 , max_steps_per_episode=4)

                
        action_classifier.save()
        likelihood_predictor.save()
        action_classifier.memory.clear()
        
 
        display(a,b,predicted,)
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