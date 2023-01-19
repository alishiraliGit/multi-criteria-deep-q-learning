import pandas as pd
import numpy as np
import os
from utils import make_train_val_test_split

def add_intermediate_rewards(data,data_rewards, data_file='MIMIC_RL.csv', save=False):
    #derive terminal state indicator
    terminal = np.zeros(data_rewards.shape[0])
    for i in range(data_rewards.shape[0]-1):
        if (i+1) > (data_rewards.shape[0]-1):
            terminal[i] = 1
        elif data_rewards['traj'][i+1] != data_rewards['traj'][i]:
            terminal[i] = 1 #this assumes a dataframe ordered by traj ID
        #else:
            #if (data[i+1,0] == 1):
                #terminal[i] = 1
    
    ####Create intermediate reward 1: One period change in SOFA Score 'o:SOFA'
    ####Create intermediate reward 2: One period change in SOFA Score binary
    ####Create intermediate reward 5: One period change in SOFA Score more than 2 binary
    reward_SOFA_1_continuous = np.zeros(data_rewards.shape[0])
    reward_SOFA_1_binary = np.zeros(data_rewards.shape[0])
    reward_SOFA_change2_binary = np.zeros(data_rewards.shape[0])
    for i in range(data_rewards.shape[0]-1):
        if (i+1) > (data_rewards.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (data_rewards['traj'][i+1] == data_rewards['traj'][i]):
                curr_SOFA = data_rewards['o:SOFA'][i]
                next_SOFA = data_rewards['o:SOFA'][i+1]
                SOFA_change = -1*(next_SOFA-curr_SOFA) #we multiply by -1 since we want to penalize increases in SOFA score
                reward_SOFA_1_continuous[i] = SOFA_change
                if SOFA_change < 0:
                    reward_SOFA_1_binary[i] = -1
                elif SOFA_change > 0:
                    reward_SOFA_1_binary[i] = 0
                else:
                    reward_SOFA_1_binary[i] = 0

                if SOFA_change <= -2:
                    reward_SOFA_change2_binary[i] = -1
                else:
                    reward_SOFA_change2_binary[i] = 0
            else:
                if data_rewards['r:reward'][i]==-1:
                    reward_SOFA_1_continuous[i] = 0
                elif data_rewards['r:reward'][i]==1:
                    reward_SOFA_1_continuous[i] = 0

    ####Create intermediate reward 3: Two period change in SOFA Score
    ####Create intermediate reward 4: Two period change in SOFA Score binary
    reward_SOFA_2_continuous = np.zeros(data_rewards.shape[0])
    reward_SOFA_2_binary = np.zeros(data_rewards.shape[0])
    for i in range(data_rewards.shape[0]-1):
        if (i+1) > (data_rewards.shape[0]-1):
            pass
        elif (i+2) > (data_rewards.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (data_rewards['traj'][i+1] == data_rewards['traj'][i]):
                curr_SOFA = data_rewards['o:SOFA'][i]
                next_SOFA = data_rewards['o:SOFA'][i+2]
                SOFA_change = -1*(next_SOFA-curr_SOFA) 
                reward_SOFA_2_continuous[i] = SOFA_change
                if SOFA_change < 0:
                    reward_SOFA_1_binary[i] = -1
                elif SOFA_change > 0:
                    reward_SOFA_1_binary[i] = 0
                else:
                    reward_SOFA_1_binary[i] = 0
            else:
                if (data_rewards['r:reward'][i]==-1 or data_rewards['r:reward'][i+1]==-1):
                    reward_SOFA_1_continuous[i] = 0
                elif (data_rewards['r:reward'][i]==1 or data_rewards['r:reward'][i+1]==1):
                    reward_SOFA_1_continuous[i] = 0
    
    ####Create intermediate reward 6: One period change in lactate levels continous
    ####Create intermediate reward 7: One period change in lactate levels binary
    reward_lactat_1_continous = np.zeros(data_rewards.shape[0])
    reward_lactat_1_binary = np.zeros(data_rewards.shape[0])
    for i in range(data_rewards.shape[0]-1):
        if (i+1) > (data_rewards.shape[0]-1):
            pass
        else:
            if (data_rewards['traj'][i+1] == data_rewards['traj'][i]):
                curr_lac = data_rewards['o:Arterial_lactate'][i]
                next_lac = data_rewards['o:Arterial_lactate'][i+1]
                lac_change = -1*(next_lac-curr_lac) 
                reward_lactat_1_continous[i] = lac_change
                if lac_change < 0:
                    reward_lactat_1_binary[i] = -1
                else:
                    reward_lactat_1_binary[i] = 0
            else:
                if data_rewards['r:reward'][i]==-1:
                    reward_lactat_1_continous[i] = 0
                elif data_rewards['r:reward'][i]==1:
                    reward_lactat_1_continous[i] = 0

    ####Create intermediate reward 8: Two period change in SOFA Score
    ####Create intermediate reward 9: Two period change in SOFA Score binary
    reward_lactat_2_continuous = np.zeros(data_rewards.shape[0])
    reward_lactate_2_binary = np.zeros(data_rewards.shape[0])
    for i in range(data_rewards.shape[0]-1):
        if (i+1) > (data_rewards.shape[0]-1):
            pass
        elif (i+2) > (data_rewards.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (data_rewards['traj'][i+1] == data_rewards['traj'][i]):
                curr_lac = data_rewards['o:Arterial_lactate'][i]
                next_lac = data_rewards['o:Arterial_lactate'][i+2]
                lac_change = -1*(next_lac-curr_lac) 
                reward_lactat_2_continuous[i] = lac_change
                if lac_change < 0:
                    reward_lactate_2_binary[i] = -1
                elif lac_change > 0:
                    reward_lactate_2_binary[i] = 0
                else:
                    reward_lactate_2_binary[i] = 0
            else:
                if (data_rewards['r:reward'][i]==-1 or data_rewards['r:reward'][i+1]==-1):
                    reward_lactat_2_continuous[i] = 0
                elif (data_rewards['r:reward'][i]==1 or data_rewards['r:reward'][i+1]==1):
                    reward_lactat_2_continuous[i] = 0

    ####Create intermediate reward 10: One period change in SOFA Score scaled
    ####Create intermediate reward 11: One period change in SOFA Score binary scaled
    ####Create intermediate reward 12: One period change in SOFA Score more than 2 binary scaled
    reward_SOFA_1_continuous_scaled = np.zeros(data_rewards.shape[0])
    reward_SOFA_1_binary_scaled = np.zeros(data_rewards.shape[0])
    reward_SOFA_change2_binary_scaled = np.zeros(data_rewards.shape[0])
    for i in range(data_rewards.shape[0]-1):
        if (i+1) > (data_rewards.shape[0]-1):
            pass
        else:
            #print(MIMIC_processing[i+1,0])
            if (data_rewards['traj'][i+1] == data_rewards['traj'][i]):
                curr_SOFA = data_rewards['o:SOFA'][i]
                next_SOFA = data_rewards['o:SOFA'][i+1]
                SOFA_change = -1*(next_SOFA-curr_SOFA)*curr_SOFA #we multiply by -1 since we want to penalize increases in SOFA score
                reward_SOFA_1_continuous_scaled[i] = SOFA_change
                if SOFA_change < 0:
                    reward_SOFA_1_binary_scaled[i] = -1*curr_SOFA
                elif SOFA_change > 0:
                    reward_SOFA_1_binary_scaled[i] = 0
                else:
                    reward_SOFA_1_binary_scaled[i] = 0

                if SOFA_change <= -2:
                    reward_SOFA_change2_binary_scaled[i] = -1*curr_SOFA
                else:
                    reward_SOFA_change2_binary_scaled[i] = 0
            else:
                if data_rewards['r:reward'][i]==-1:
                    reward_SOFA_1_continuous_scaled[i] = 0
                elif data_rewards['r:reward'][i]==1:
                    reward_SOFA_1_continuous_scaled[i] = 0
        
    ## New rewards
    sparse_reward = data_rewards['r:reward']

    data['terminal'] = terminal
    data['sparse_90d_rew'] = sparse_reward

    data['Reward_SOFA_1_continous'] = reward_SOFA_1_continuous
    data['Reward_SOFA_1_binary'] = reward_SOFA_1_binary
    data['Reward_SOFA_2_continous'] = reward_SOFA_2_continuous
    data['Reward_SOFA_2_binary'] = reward_SOFA_2_binary
    data['Reward_SOFA_change2_binary'] = reward_SOFA_change2_binary

    data['Reward_lac_1_continous'] = reward_lactat_1_continous
    data['Reward_lac_1_binary'] = reward_lactat_1_binary
    data['Reward_lac_2_continous'] = reward_lactat_2_continuous
    data['Reward_lac_2_binary'] = reward_lactate_2_binary

    data['Reward_SOFA_1_continous_scaled'] = reward_SOFA_1_continuous_scaled
    data['Reward_SOFA_1_binary_scaled'] = reward_SOFA_1_binary_scaled
    data['Reward_SOFA_change2_binary_scaled'] = reward_SOFA_change2_binary_scaled

    #Save the augmented MIMIC table
    if save:
        data.to_csv(data_file[:-4] +'_incl_rewards.csv')
    
    return data

# 80% training data + 5% validation data + 15% test data
train_fraction = 0.8
validation_fraction = 0.05

#data_file = r"./data/sepsis_mimiciii/sepsis_final_data_K1.csv"

data_file = r"./data/sepsis_mimiciii/sepsis_final_data_withTimes.csv"
#data_file = r"./data/sepsis_mimiciii/sepsis_final_data_RAW_withTimes.csv"
data_file_rewards = r"./data/sepsis_mimiciii/sepsis_final_data_RAW_withTimes.csv"

#Check out data
data = pd.read_csv(data_file)
data_rewards = pd.read_csv(data_file)

#############################
# Added content
#############################

print(data.head(10))
print(data.columns)
print(data.describe())

#Add rewards to dataset
add_rewards = True

if add_rewards:
    data = add_intermediate_rewards(data=data, data_rewards=data_rewards,data_file=data_file,save=False)
    #print(data[['traj','r:reward','Reward_SOFA_1_continous','Reward_lac_1_continous','terminal']])
    print(data.columns)
    
#############################
# Original content
#############################

# remove extra columns
save = False
save = True
for item in ['o:input_total', 'o:input_4hourly', 'o:max_dose_vaso']:
    if item in data.columns:
        data.drop(labels=[item], axis=1, inplace=True)
        save = True
if save:
    os.rename(data_file, data_file[:-4] + '_fullcolumns.csv')
    data.to_csv(os.path.join(data_file))  # overwrite with new data_file


# will save the split data in the same folder as data_file
print("Processing ...")
make_train_val_test_split(filename=data_file, train_frac=train_fraction, val_frac=validation_fraction, make_test=True)
print("Done.")
