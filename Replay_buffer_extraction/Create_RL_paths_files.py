"""
This script will load the created transition dataframe and convert it into a list of paths
I.e.
1. Combine all observations of a given trajectory in a path dictionary
2. Store all of the paths in a list
3. Save that list as a pkl file to be later imported into the replay buffer

(Tasks to come in other script - implement offline case for current pipeline and run using created path file)
"""

import pickle 
import numpy as np 
import pandas as pd 
import os

import math 
import datetime 

#From hw5 utils
def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def create_path_file(unique_trajectories,REWARD_TO_INCLUDE,mimictable_transitions):
    paths = []
    image_obs = []
    
    for id in unique_trajectories:
        df = mimictable_transitions[mimictable_transitions["icustayid"]==id]
        obs, acs, next_obs, terminals, rewards = [], [], [], [], []
        for row in df.to_dict(orient='records'):
            obs.append(row['state'])
            acs.append(row['action'])
            next_obs.append(row['next_state'])
            terminals.append(row['terminal'])
            rewards.append(row[REWARD_TO_INCLUDE])
        path = Path(obs, image_obs, acs, rewards, next_obs, terminals)
        paths.append(path)

    file_loc = os.path.join(os.path.join('Replay_buffer_extraction'), f'Paths_{REWARD_TO_INCLUDE}.pkl')
    
    with open(file_loc, 'wb') as f:
       pickle.dump(paths, f)

def create_path_file_all(unique_trajectories,mimictable_transitions):
    paths = []
    image_obs = []
    
    for id in unique_trajectories:
        df = mimictable_transitions[mimictable_transitions["icustayid"]==id]
        obs, acs, next_obs, terminals = [], [], [], []
        r1, r2, r3, r4 = [], [], [], []
        r5, r6, r7, r8 = [], [], [], []
        r9, r10, r11 = [], [], []
        for row in df.to_dict(orient='records'):
            obs.append(row['state'])
            acs.append(row['action'])
            next_obs.append(row['next_state'])
            terminals.append(row['terminal'])
            r1.append(row['sparse_90d_rew'])
            r2.append(row['Reward_matrix_paper'])
            r3.append(row['Reward_SOFA_1_continous'])
            r4.append(row['Reward_SOFA_1_binary'])
            r5.append(row['Reward_SOFA_2_continous'])
            r6.append(row['Reward_SOFA_2_binary'])
            r7.append(row['Reward_SOFA_change2_binary'])
            r8.append(row['Reward_lac_1_continous'])
            r9.append(row['Reward_lac_1_binary'])
            r10.append(row['Reward_lac_2_continous'])
            r11.append(row['Reward_lac_2_binary'])
        path = Path_all(obs, image_obs, acs, next_obs, terminals, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11)
        paths.append(path)
    
    file_loc = os.path.join(os.path.join('Replay_buffer_extraction'), f'Paths_all_rewards.pkl')

    with open(file_loc, 'wb') as f:
       pickle.dump(paths, f)


def Path_all(obs, image_obs, acs, next_obs, terminals, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "sparse_90d_rew" : np.array(r1, dtype=np.float32),
            "Reward_matrix_paper" : np.array(r2, dtype=np.float32),
            "Reward_SOFA_1_continous" : np.array(r3, dtype=np.float32),
            "Reward_SOFA_1_binary" : np.array(r4, dtype=np.float32),
            "Reward_SOFA_2_continous" : np.array(r5, dtype=np.float32),
            "Reward_SOFA_2_binary" : np.array(r6, dtype=np.float32),
            "Reward_SOFA_change2_binary" : np.array(r7, dtype=np.float32),
            "Reward_lac_1_continous" : np.array(r8, dtype=np.float32),
            "Reward_lac_1_binary" : np.array(r9, dtype=np.float32),
            "Reward_lac_2_continous" : np.array(r10, dtype=np.float32),
            "Reward_lac_2_binary" : np.array(r11, dtype=np.float32)}

if __name__ == '__main__': 

    #load the full MIMIC_table with rewards
    mimictable_full = pd.read_csv('C:/Users/alexa/Alex/Studium/01_UC_Berkeley/01_Courses/Fall22/CS285_Deep_RL/Project/Codebase/py_ai_clinician/MIMICtable_plus_SanSTR.csv')

    #load the MIMIC table only containing state action next state, rewards
    mimictable_transitions = pd.read_csv('C:/Users/alexa/Alex/Studium/01_UC_Berkeley/01_Courses/Fall22/CS285_Deep_RL/Project/Codebase/py_ai_clinician/MIMICtable_transitions.csv')

    #This are the relevant columns of the dataset
    #['Unnamed: 0', 'bloc', 'icustayid', 'charttime', 'state', 'action',
    #   'next_state', 'terminal', 'sparse_90d_rew', 'Reward_matrix_paper',
    #   'Reward_SOFA_1_continous', 'Reward_SOFA_1_binary',
    #   'Reward_SOFA_2_continous', 'Reward_SOFA_2_binary',
    #   'Reward_SOFA_change2_binary', 'Reward_lac_1_continous',
    #   'Reward_lac_1_binary', 'Reward_lac_2_continous', 'Reward_lac_2_binary']

    unique_trajectories = list(mimictable_transitions.icustayid.unique())
    
    #IDEA 1: one reward per path file, multiple Path files

    reward_names = ['sparse_90d_rew', 'Reward_matrix_paper', 'Reward_SOFA_1_continous', 'Reward_SOFA_1_binary',
       'Reward_SOFA_2_continous', 'Reward_SOFA_2_binary',
       'Reward_SOFA_change2_binary', 'Reward_lac_1_continous',
       'Reward_lac_1_binary', 'Reward_lac_2_continous', 'Reward_lac_2_binary']
    
    for reward_to_include in reward_names:
        print(f'Creating Paths file for {reward_to_include}')
        create_path_file(unique_trajectories,reward_to_include,mimictable_transitions)
        print('saved paths file')
    
    #IDEA 2: One Path file with all of the rewards
    print(f'Creating Paths file for all rewards')
    create_path_file_all(unique_trajectories,mimictable_transitions)
    print('saved paths file')








