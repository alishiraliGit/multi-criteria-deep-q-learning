"""
# Construct rewards based on data in dataframe --> done in data_process script to be re-run for normalized data

# Load data or store filepath of data in correct format for state constructor 
# (trick set complete file to train file such that we get states for the whole dataset)

# Construct states for the whole dataset that is in the correct format

# Concatenate rewards to dataset with states  

"""

import os
import click
import yaml
import numpy as np
import pandas as pd
import pickle
import torch
from state_construction import StateConstructor
from rl import RL
from experiment import DQNExperiment
from utils import DataLoader

from tqdm import tqdm

def load_best_sc_network(params, rng):
    # NOTE: returned SC-Network will need to either re-load data or load some new test data
    store_path = os.path.join(params["folder_location"], params["folder_name"])  # this is `sc_network.store_path` if a SC-Network is loaded with params 
    # Initialize the SC-Network
    sc_network = StateConstructor(train_data_file=params["train_data_file"], validation_data_file=params["validation_data_file"], 
                            minibatch_size=params["minibatch_size"], rng=rng, device=params["device"], save_for_testing=params["save_all_checkpoints"],
                            sc_method=params["sc_method"], state_dim=params["embed_state_dim"], sc_learning_rate=params["sc_learning_rate"], 
                            ais_gen_model=params["ais_gen_model"], ais_pred_model=params["ais_pred_model"], sc_neg_traj_ratio=params["sc_neg_traj_ratio"], 
                            folder_location=params["folder_location"], folder_name=params["folder_name"], 
                            num_actions=params["num_actions"], obs_dim=params["obs_dim"])
    sc_network.reset_sc_networks()
    # Provide SC-Network with the pre-trained parameter set
    sc_network.load_model_from_checkpoint(checkpoint_file_path=os.path.join(store_path, "ais_checkpoints", "checkpoint_best.pt"))
    return sc_network

def create_path_file_all(unique_trajectories,mimictable_transitions, train_data_encoded):
    paths = []
    image_obs = []
    
    for id in tqdm(unique_trajectories):
        df = mimictable_transitions[mimictable_transitions["traj"]==id]
        obs, acs, next_obs, terminals = [], [], [], []
        r1, r2, r3, r4 = [], [], [], []
        r5, r6, r7, r8 = [], [], [], []
        r9, r10, r11 = [], [], []
        r12, r13, r14 = [], [], []
        for row in df.to_dict(orient='records'):
            #obs.append(row['state'])
            #acs.append(row['action'])
            #next_obs.append(row['next_state'])
            terminals.append(row['terminal'])
            r1.append(row['sparse_90d_rew'])
            #r2.append(row['Reward_matrix_paper'])
            r3.append(row['Reward_SOFA_1_continous'])
            r4.append(row['Reward_SOFA_1_binary'])
            r5.append(row['Reward_SOFA_2_continous'])
            r6.append(row['Reward_SOFA_2_binary'])
            r7.append(row['Reward_SOFA_change2_binary'])
            r8.append(row['Reward_lac_1_continous'])
            r9.append(row['Reward_lac_1_binary'])
            r10.append(row['Reward_lac_2_continous'])
            r11.append(row['Reward_lac_2_binary'])

            r12.append(row['Reward_SOFA_1_continous_scaled'])
            r13.append(row['Reward_SOFA_1_binary_scaled'])
            r14.append(row['Reward_SOFA_change2_binary_scaled'])
        
        #Insert zeros for 'Reward_matrix_paper', this is a legacy reward from the Komorowski paper that is not used in our work
        r2 = [0]*len(r1) 
        
        obs = train_data_encoded['traj'][id]['s']
        acs = train_data_encoded['traj'][id]['actions']
        next_obs = list(train_data_encoded['traj'][id]['s'][1:])

        #TODO raise this issue of the absorbing states
        # Code absorbing states for death and survival, numerical choice of 5 was arbitrary
        if sum(r1) == 1:
            absorbing_state_survival = [5]*64
            next_obs.append(absorbing_state_survival)
        elif sum(r1) == -1:
            absorbing_state_death = [-5]*64
            next_obs.append(absorbing_state_death)

        path = Path_all(obs, image_obs, acs, next_obs, terminals, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14)
        paths.append(path)
    
    #file_loc = os.getcwd() + 'Paths_all_rewards_sc.pkl'
    file_loc = '/Users/alexa/Alex/Studium/01_UC_Berkeley/01_Courses/Fall22/CS285_Deep_RL/Project/Codebase/Paths_all_rewards_sc.pkl'
    #file_loc = os.path.join(os.getcwd(), f'Paths_all_rewards_sc.pkl')

    #file_loc = os.path.join(os.path.join('Replay_buffer_extraction'), f'Paths_all_rewards.pkl')
    #file_loc = 'Paths_all_rewards_sc.pkl'

    #if not os.path.exists(file_loc):
            #os.mkdir(file_loc)

    with open(file_loc, 'wb') as f:
       pickle.dump(paths, f)


def Path_all(obs, image_obs, acs, next_obs, terminals, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14):
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
            "Reward_lac_2_binary" : np.array(r11, dtype=np.float32),
            "Reward_SOFA_1_continous_scaled" : np.array(r12, dtype=np.float32),
            "Reward_SOFA_1_binary_scaled" : np.array(r13, dtype=np.float32),
            "Reward_SOFA_change2_binary_scaled" : np.array(r14, dtype=np.float32)
            }

if __name__ == '__main__':
    ### Consider plugging all of this into a run function later

    #initialize config file folder
    #folder = r'Users/alexa/Alex/Studium/01_UC_Berkeley/01_Courses/Fall22/CS285_Deep_RL/Project/Codebase/med-deadend'
    folder = os.getcwd()

    #load parameters for state_network from config file
    folder = os.path.abspath(folder)
    with open(os.path.join(folder, "config_sepsis_state_extraction.yaml")) as f:
        params = yaml.safe_load(f)
    
    #print out loaded params
    print('Parameters ')
    for key in params:
        print(key, params[key])
    print('=' * 30)

    #get random seeds for reproducability
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])
    rng = np.random.RandomState(params['random_seed'])

    # Initialize and load the pre-trained parameters for the SC-Network
    sc_network = load_best_sc_network(params, rng)  # note that the loaded SC-Network has no data inside
    params["used_checkpoint_for_rl"] = "checkpoint_best.pt"

    sc_network.load_mk_train_validation_data()
    print("Train data ...")
    train_data_encoded = sc_network.encode_data(sc_network.train_data_trajectory)
    print("Validation data ...")
    validation_data_encoded = sc_network.encode_data(sc_network.validation_data_trajectory)

    #inspect the encoded dataset
    input_dataframe = pd.read_csv('data/sepsis_mimiciii/sepsis_final_data_withTimes.csv')
    print(input_dataframe.columns)
    print(input_dataframe[['traj','a:action']].head(15))


    print(type(train_data_encoded))
    print(train_data_encoded.keys())
    print(train_data_encoded['traj'][1]['actions'])
    print(type(train_data_encoded['traj']))
    print(train_data_encoded['obs_cols'][:5])

    #check that the ids are the same across df and encoded data
    """
    traj_test = [1,22,55,1278,324,77,2345,975,479]
    for id in traj_test:
        print('reference actions')
        print(input_dataframe[input_dataframe['traj']==id]['a:action'])
        print('encoded data actions')
        print(train_data_encoded['traj'][id]['actions']) 
        print(train_data_encoded['traj'][id]['s']) 
        print(list(input_dataframe[input_dataframe['traj']==id]['a:action'] == train_data_encoded['traj'][id]['actions']))
    """

    unique_trajectories = list(input_dataframe['traj'].unique())
    print(max(train_data_encoded['traj'].keys()))
    #TODO decide whether we from here can package data into a path file, mainly dependent on order
    #TODO other option is to adapt the trajectory data creation folder in the sc_network object
    print('Creating path file with all rewards')
    create_path_file_all(unique_trajectories,input_dataframe,train_data_encoded)
    print('saved paths file')

    
    