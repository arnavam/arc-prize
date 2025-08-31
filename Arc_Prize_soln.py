from dl_models.BaseDQN import BaseDQN
import numpy as np
import time 
import os
from dsl import ALL_ACTIONS ,SHIFT_ACTIONS,TRANSFORM_ACTIONS
from helper import find_objects , extract_target_region
from helper_arc import  loader , display
from dl_models.Feature_Extractor import FeatureExtractor
from dl_models.DQNAction_Classifier import DQN_Classifier
from dl_models.ReinLikelihood import Likelihood
import torch
import copy
torch.autograd.set_detect_anomaly(True)
from helper_env import  placement , place_object
from helper_env import matrix_similarity
import logging

import logging
# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/Arc_Prize_soln.log', mode='w')
# handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False


# Initialized parameters
#----------------------------------
action_names = list(ALL_ACTIONS.keys())
shift_actions = list(SHIFT_ACTIONS.keys())
transform_actions=list(TRANSFORM_ACTIONS.keys())
num_actions = len(action_names)


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
        return np.argmax(scores)

    def update_arm(self, arm, reward):
        self.n[arm] += 1
        self.sum_r[arm] += reward
        self.sum_sq[arm] += reward**2

    def mark_as_solved(self, arm):
        print(f"Bandit: Marking example {arm} as solved.")
        self.solved_status[arm] = True
#Initialized functions
#---------------------------------------------------------------------


def Arc_Prize_Solver(examples,load,save,max_iterations=100, max_steps_per_episode=4):


    ft1 = FeatureExtractor(input_channels=1)
    action_classifier = DQN_Classifier(feature_extractor=ft1, num_actions=num_actions,no_of_inputs=3)

    ft2 = FeatureExtractor(input_channels=1)
    likelihood_predictor = Likelihood(feature_extractor= ft2, no_of_outputs=1,no_of_inputs=3)

    Placer = None #DQN_Solver(ft,len(examples[0]['output']),3)

    if load == True:
        action_classifier.load()
        likelihood_predictor.load()
    
    logger.debug(f'output shape & no of examples{len(examples[0]['output']),len(examples)}')

    num_examples = len(examples)

    bandit = Example_Chooser(num_examples)

    objects = None
    obj_list={}
    count =0
    for  _ in range(max_iterations):
        count+=1
        idx = bandit.select_example()
        if idx == -1:
            if  save ==True:
                action_classifier.save()
                likelihood_predictor.save()
            return example['predicted_grid'] , True
        
        logger.debug(f'count{idx,count}')
        example = examples[idx]

        input_grid = np.array(example['input'])
        target_grid = np.array(example['output'])

        logger.debug(f'input grid: {input_grid}')
        logger.debug(f'target grid: {target_grid}')
        solved=0
        if idx not in obj_list:
            objects = find_objects(input_grid)
            obj_list[idx]=objects

            predicted_grid = np.zeros_like(target_grid)
            example['predicted_grid'] = predicted_grid 
            logger.debug('new_predicted_grid')
        else:
            objects=obj_list[idx]   
            predicted_grid=example['predicted_grid']

        logger.debug(f"'predicted_grid',\n{predicted_grid},{type(predicted_grid)}")

        new_grid = predicted_grid
        old_reward=0
        sim_score=0
        for step in range(max_steps_per_episode):
            new_grid ,new_reward= find_solution(new_grid,likelihood_predictor,action_classifier,Placer, target_grid,objects)
            
            sim_score += new_reward-old_reward
            old_reward=new_reward
                    
            if np.array_equal(new_grid, target_grid):
                #remove the current example from the iterations
                solved +=1 
                bandit.mark_as_solved(idx)
                logger.debug(f'{idx} win no {solved} :\n{predicted_grid}')
                #if no more to remove say we solved everyone and use maybe testing example to test the solution
                break

            example['predicted_grid']= new_grid

            print(sim_score)
            action_classifier.update_policy()
            likelihood_predictor.update_policy()
            bandit.update_arm(idx, sim_score)
        

    logger.info("No solution found within iterations")
    if  save ==True:
        action_classifier.save()
        likelihood_predictor.save()
    return predicted_grid , False  # No solution found

#----------------------------------------------------------------------------------------------------



def find_solution(old_predicted_grid, likelihood_predictor,action_classifier, Placer_, target_grid,objects):

 
    obj_idx,obj_prob = likelihood_predictor.select_action([old_predicted_grid, objects, target_grid])
    obj_info  = objects[obj_idx]

    action_idx ,(x,y) = action_classifier.select_action([old_predicted_grid,  obj_info['grid'], obj_info['position'],target_grid])

    pos_values = [int(x * target_grid.shape[1]), int(y * target_grid.shape[0])]

    logger.debug(f'target_shape: {target_grid.shape} , new pos_values: {pos_values}')

    action = action_names[action_idx]
    is_place_action = False
    new_obj_info = copy.deepcopy(obj_info)

    if action == 'place' or obj_info['placed'] == False:


        new_obj_info['placed']=True
        is_place_action = True
        
        new_obj_info['position'] = pos_values

        objects.append(new_obj_info)
        
        # new_predicted_grid=place_object(old_predicted_grid,new_obj_info['grid'],pos_values)
        obj_info=copy.deepcopy(new_obj_info)

    elif action in transform_actions:
        
        new_obj_info['grid']=ALL_ACTIONS[action](obj_info['grid'])


    elif action in shift_actions:
        new_obj_info['position']=ALL_ACTIONS[action](obj_info['position'])

    new_predicted_grid =placement(old_predicted_grid, obj_info, new_obj_info, background=0)


    reward = matrix_similarity(new_obj_info['grid'],extract_target_region(target_grid,new_obj_info))
    logger.debug(f"{action}: \n{obj_info['grid']}\n{new_predicted_grid}")

    if new_predicted_grid is None:
        logger.debug('new_matrix')
        new_predicted_grid = old_predicted_grid
        reward = 0
    else : logger.debug('performed')


    h, w = target_grid.shape[:2]
    norm_pos = (
        new_obj_info['position'][0] / w,
        new_obj_info['position'][1] / h
    )
    logger.debug(f'new obj position{norm_pos}')

    action_classifier.store_experience(
        state=(old_predicted_grid, obj_info['grid'], obj_info['position']),

        action=action_idx,
        reward=reward,
        next_state=(new_predicted_grid, new_obj_info['grid'], new_obj_info['position']),
        true_position=norm_pos,
        is_place_action=is_place_action
    )


 
    reward= matrix_similarity(new_predicted_grid,target_grid)
    likelihood_predictor.store_experience(obj_prob,reward)

    return new_predicted_grid ,reward


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
        logger.debug(f"Processing task {case_id} with {len(examples)} examples")

        predicted , success = Arc_Prize_Solver(examples,load=True,save=False ,max_iterations=100)
        display(a,b,predicted)
        if success:
            a = task['train'][0]['input']
            b= task['train'][0]['output']
            
            print(f"Task {case_id} solved")
        else:
            print(f"Task {case_id} not solved")
