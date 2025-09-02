from rl_models.BaseDQN import BaseDQN
import numpy as np
import time 
import os
from dsl import COMB ,SHIFT,TRANSFORM
from dsl2 import find_objects , extract_target_region
from A_arc import  loader , display 
from rl_models.reinforce import FeatureExtractor
from rl_models.DQNAction_Classifier import DQN_Classifier
from rl_models.ReinLikelihood import Likelihood
import torch
import torch
torch.autograd.set_detect_anomaly(True)
from env import  placement
from env import matrix_similarity
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to predicted_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)

# Initialized parameters
#----------------------------------
action_names = list(COMB.keys())
Shift = list(SHIFT.keys())
Transform=list(TRANSFORM.keys())
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
                logging.debug(f"Arm {i}: Mean={mean:.4f}, Std={std:.4f} -> Score={scores[i]:.4f}")           
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
    likelihood_predictor = Likelihood(feature_extractor= ft2, output_dim=1,no_of_inputs=3)

    Placer = None #DQN_Solver(ft,len(examples[0]['output']),3)

    if load == True:
        # action_classifier.load()
        likelihood_predictor.load()
    
    logging.debug(f'output shape & no of examples{len(examples[0]['output']),len(examples)}')

    num_examples = len(examples)

    bandit = Example_Chooser(num_examples)

    objects = None
    obj_list={}
    count =0
    for iteration in range(max_iterations):
        count+=1
        idx = bandit.select_example()
        if idx == -1:
            if  save ==True:
                action_classifier.save()
                likelihood_predictor.save()
            return example['predicted_grid'] , True
        
        logging.debug(f'count{idx,count}')
        example = examples[idx]

        input_grid = np.array(example['input'])
        target_grid = np.array(example['output'])

        logging.debug(f'input grid: {input_grid}')
        logging.debug(f'target grid: {target_grid}')
        solved=0
        if idx not in obj_list:
            objects = find_objects(input_grid)
            obj_list[idx]=objects

            predicted_grid = np.zeros_like(target_grid)
            example['predicted_grid'] = predicted_grid 
        else:
            objects=obj_list[idx]   
            predicted_grid=example['predicted_grid']

        logging.debug(f"'predicted_grid',{predicted_grid},{type(predicted_grid)}")

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
                logging.debug(f'{idx} win no {solved} :{predicted_grid}')
                #if no more to remove say we solved everyone and use maybe testing example to test the solution
                break

            example['predicted_grid']= predicted_grid

            print(sim_score)
            action_classifier.update_policy()
            likelihood_predictor.update_policy()
            bandit.update_arm(idx, sim_score)
        

    logging.info("No solution found within iterations")
    if  save ==True:
        action_classifier.save()
        likelihood_predictor.save()
    return predicted_grid , False  # No solution found

#----------------------------------------------------------------------------------------------------



def find_solution(old_predicted_grid, likelihood_predictor,action_classifier, Placer_, target_grid,objects):

 
    idx,prob1 = likelihood_predictor.select_action([old_predicted_grid, objects, target_grid])
    obj_info  = objects[idx]

    action_idx ,(x,y)= action_classifier.select_action([old_predicted_grid,  obj_info['grid'], obj_info['position'],target_grid])

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
    
    elif func in Transform:
        
        new_obj_info['grid']=COMB[func](obj_info['grid'])

    elif func in Shift:
        new_obj_info['position']=COMB[func](obj_info['position']) 

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

    for case_id in ids:
        count +=1
        if count ==2:
            break

        task = train[case_id]
        examples = task['train']  # Assume each task has a 'train' list of examples
        print(examples)
        logging.debug(f"Processing task {case_id} with {len(examples)} examples")

        predicted , success = Arc_Prize_Solver(examples,load=False,save=True ,max_iterations=100)

        predicted , success = Arc_Prize_Solver(examples,load=False,save=False ,max_iterations=100)
        display(a,b,predicted)
        if success:
            a = task['train'][0]['input']
            b= task['train'][0]['output']
            display(a,b,predicted)
            print(f"Task {case_id} solved")
        else:
            print(f"Task {case_id} not solved")