import pickle

path = "/Users/alexa/Alex/Studium/01_UC_Berkeley/01_Research_Projects/mul-dqn/multi-criteria-deep-q-learning/data/v6_var1c_1_offline_pruned_cmdqn_alpha20_sparse_eval_MIMIC-Continuous/metrics.pkl"

with open(path, 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)  # print the loaded dictionary