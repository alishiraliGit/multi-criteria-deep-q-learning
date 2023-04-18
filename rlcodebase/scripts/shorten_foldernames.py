import os
import re

#Please fix the directory path below. I am getting the error: The system cannot find the path specified: '~/Downloads/phase2_to_Alex'
#directory = '../../../../../Downloads/phase2_to_Alex'
directory = '~/Downloads/phase1_to_Alex/'
directory = os.path.expanduser(directory)

print(directory)

#recall input data structure
#v6_var1c_1_offline_pruned_cmdqn_lr[1]1e-05_tuf[1]1000_r[1]10_lr[2]1e-04_tuf[2]8000_cql0.001_alpha20_sparse_MIMIC-Continuous_12-04-2023_21-16-12
#v6_var1c_5_offline_cmdqn_lr1e-05_tuf1000_cql0.001_alpha20_r10_MIMIC-MultiContinuousReward_13-04-2023_11-40-09

#loop through directory to rename folders
for foldername in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, foldername)):
        new_foldername = re.sub('_cql0.001_', '_', foldername)
        #new_foldername = re.sub('_cmdqn_lr\d+e-\d+_tuf\d+_', '_cmdqn_', foldername)
        #new_foldername = re.sub('_lr1e-05_tuf1000_cql0.001_', '_', foldername)
        #new_foldername = re.sub('_r10_', '_', foldername)
        #new_foldername = re.sub('_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', new_foldername)
        #new_foldername = re.sub('-Continous', '', new_foldername)
        os.rename(os.path.join(directory, foldername), os.path.join(directory, new_foldername))
