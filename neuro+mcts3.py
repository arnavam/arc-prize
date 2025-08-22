from RL_alg.BaseDQN import BaseDQN
import numpy as np
import time 
import os
from dsl import find_objects , COMB ,ACTIONS,PRIMITIVE
from A_arc import  loader , display
from RL_alg.reinforce import FeatureExtractor
from RL_alg.DQNAction_Classifier import DQN_Solver_MultiHead
from RL_alg.DQNLikelihood import Likelihood

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
action_names = list(COMB.keys())
names1 = list(ACTIONS.keys())
names2=list(PRIMITIVE.keys())

num_actions = len(action_names)

# Initialized models
ft = FeatureExtractor(input_channels=1)
neuro_classifer = DQN_Solver_MultiHead(ft, num_actions,3)
Likeliehood = Likelihood(ft, 1,3)

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

def find_solution(current_grid, Placer, target_grid,objects):
    prob = 0
    obj_info = objects[0]
    for obj in objects:
        # Likeliehood predicts the relevance of an object
        prob1 = Likeliehood.select_action([current_grid, obj['grid'], target_grid])
        # print('prob',prob1)
        if prob < prob1:
            prob = prob1
            obj_info = obj


    # neuro_classifer predicts the best action for the chosen object
    action_idx ,pos_values= neuro_classifer.select_action([current_grid,  obj_info['grid'], obj_info['position'],target_grid])
    # Apply the action using Placer
    pos_values = [int(pos_values[0] * target_grid.shape[1]), int(pos_values[1] * target_grid.shape[0])]
    print('target_shape',target_grid.shape)
    print('new pos_values',pos_values)
    func= action_names[action_idx]

    if func == 'idle' or obj_info['placed'] == False:
        # print('placed')
        new_obj_info =obj_info.copy()
        obj_info['placed']=True
        new_obj_info['position'] = pos_values
        
        objects.append(new_obj_info)
        obj_info = new_obj_info
        new_grid =placement(current_grid, obj_info, new_obj_info, background=0)
        print('old output_grid',current_grid)
        print('obj_info',obj_info['grid'])
        print('pos',obj_info['position'])
        print('new output_grid',new_grid)
    elif func in names2:
        
        new_obj_info =obj_info.copy()
        new_obj_info['grid']=COMB[func](obj_info['grid'])
        new_grid =placement(current_grid, obj_info, new_obj_info, background=0)

    elif func in names1:
        new_obj_info = obj_info.copy()
        new_obj_info['position']=COMB[func](obj_info['position'])   
        new_grid =placement(current_grid, obj_info, new_obj_info, background=0)
    if new_grid is None:
        new_grid = current_grid
    return new_grid

def Gaussian_multi_armbandit(examples, max_iterations=1000, max_steps_per_episode=10):
    """
    Gaussian multi-armed bandit for selecting training examples.
    Args:
        examples: List of training examples, each with 'input' and 'output' grids.
        max_iterations: Maximum number of bandit iterations.
        max_steps_per_episode: Maximum steps to try for each example.
    Returns:
        solution: The solution that works for all examples, or None if not found.
    """
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


        for step in range(max_steps_per_episode):
            new_grid = find_solution(current_grid, Placer, target_grid,objects)
            if np.array_equal(new_grid, target_grid):
                break
            current_grid = new_grid

        example['current_grid']= current_grid
        similarity = matrix_similarity(new_grid, target_grid)
        bandit.update_arm(idx, similarity)
        
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