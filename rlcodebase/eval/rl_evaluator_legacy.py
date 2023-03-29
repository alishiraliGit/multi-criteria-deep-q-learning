import pickle
import os

import numpy as np
import torch
import gym
from gym import wrappers
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import rlcodebase.envs.mimic_utils
from rlcodebase.envs.gym_utils import register_custom_envs
from rlcodebase.pruners.base_pruner import BasePruner
from rlcodebase.infrastructure.utils import pytorch_utils as ptu


class RLEvaluatorLegacy(object):

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params, create logger
        self.params = params
        self.log_dir = self.params['logdir']

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        # To be set later in run_training_loop
        self.total_envsteps = None
        self.start_time = None

        #############
        # ENV
        #############
        if not self.params['offline']:
            # Make the gym environment
            register_custom_envs()

            self.env = gym.make(self.params['env_name'])

            # Added by Ali
            if params['env_name'] == 'LunarLander-Customizable' and params['env_rew_weights'] is not None:
                self.env.set_rew_weights(params['env_rew_weights'])

            self.episode_trigger = lambda episode: False

            if 'env_wrappers' in self.params:
                # These operations are currently only for Atari envs
                self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
                self.env = ReturnWrapper(self.env)
                self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], 'gym'),
                                                episode_trigger=self.episode_trigger)
                self.env = params['env_wrappers'](self.env)
                self.mean_episode_reward = -float('nan')
                self.best_mean_episode_reward = -float('inf')

            self.env.seed(seed)

            # Import plotting (locally if 'obstacles' env)
            if not (self.params['env_name'] == 'obstacles-rlcodebase-v0'):
                import matplotlib
                matplotlib.use('Agg')

            # Maximum length for episodes
            self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

            # Is this env continuous, or discrete?
            discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

            # Are the observations images?
            img = len(self.env.observation_space.shape) > 2

            self.params['agent_params']['discrete'] = discrete

            # Observation and action sizes
            ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
            ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
            self.params['agent_params']['ac_dim'] = ac_dim
            self.params['agent_params']['ob_dim'] = ob_dim


    def run_evaluation_loop(self, n_iter, opt_policy, eval_pruner: BasePruner, buffer_path, pruning_critic=None, pruned_policy=None):
        # TODO: Hard-coded
        print_period = 1

        opt_actions = []
        policy_actions = []
        pruned_actions = []
        action_flags = []

        all_q_values = []
        all_rtgs = []

        


        #This are the biomarkers
         #['gender', 'age', 'elixhauser', 're_admission', 'died_in_hosp', 'died_within_48h_of_out_time', 
         # 'mortality_90d', 'delay_end_of_record_and_discharge_or_death', 'Weight_kg', 'GCS', 'HR', 'SysBP', 
         # 'MeanBP', 'DiaBP', 'RR', 'SpO2', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 
         # 'BUN', 'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 
         # 'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 
         # 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'mechvent', 'Shock_Index', 'PaO2_FiO2', 'median_dose_vaso', 
         # 'max_dose_vaso', 'input_total', 'input_4hourly', 'output_total', 'output_4hourly', 
         # 'cumulated_balance', 'SOFA', 'SIRS']

         #Fatemi uses: Arterial_lactate, SOFA, SIRS, BUN, HR, 
         # DiaBP, INR, MeanBP, RR, SpO2, SysBP, Temp_C, GCS 
         # mechvent, paO2, PTT

         # ['SOFA', 'SIRS', 'Arterial_lactate', 'Arterial_pH', 'BUN', 'HR', 'DiaBP', 'INR', 'MeanBP', 'RR', 'SpO2',
         # 'SysBP', 'Temp_C', 'GCS', 'mechvent', 'paO2', 'paCO2']

        sofa, sirs, art_lactate, art_ph, bun, hr, diabp  = [], [], [], [], [], [], []
        inr, meanbp, rr, spo2, sysbp, temp_c, gcs, mechvent, pao2, paco2  = [], [], [], [], [], [], [], [], [], []

        if buffer_path is not None:
            # Load replay buffer data
            with open(self.params['buffer_path'], 'rb') as f:
                all_paths = pickle.load(f)
            if self.params['env_name'] == 'MIMIC':
                all_paths = rlcodebase.envs.mimic_utils.format_reward(all_paths, weights=self.params['env_rew_weights'])
            elif self.params['env_name'] == 'MIMIC-Continuous':
                all_paths = rlcodebase.envs.mimic_utils.format_reward(all_paths, weights=self.params['env_rew_weights'], continuous=True)
            else:
                raise Exception('Invalid env_name!')
            # Evaluate on 5% hold-out set
            _, paths = train_test_split(all_paths, test_size=0.05, random_state=self.params['seed'])

        # We run the loop only once in the MIMIC setting since we do not sample trajectories
        if self.params['offline']:
            n_iter = 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # Collect trajectories
            if buffer_path is None:
                paths, envsteps_this_batch = utils.sample_trajectories(self.env, opt_policy,
                                                                       min_timesteps_per_batch=self.params[
                                                                           'batch_size'],
                                                                       max_path_length=self.params['ep_len'],
                                                                       render=False)

            # Get optimal actions and pruned_actions
            for path in tqdm(paths):
                # get actions
                opt_actions.append(path['action'].astype(int).tolist())

                # get pareto action sets per traj
                pareto_actions = eval_pruner.get_list_of_available_actions(path['observation'])
                pruned_actions.append(pareto_actions)

                # Flag if action not in pareto-set
                flags = [1 if path['action'][i] in pareto_actions[i] else 0 for i in range(len(path['action']))]
                action_flags.append(flags)

                # Get reward to go (and transform to mortality indicator, assumes Gamma == 1)
                # TODO
                if self.params['env_name'].startswith('LunarLander'):
                    reward_tag = 'reward'
                else:
                    reward_tag = 'sparse_90d_rew'

                rtg_n = utils.discounted_cumsum(path[reward_tag],
                                                self.params['gamma']) / 100  # to make this a mortality indicator
                rtg_n = (rtg_n + 1) / 2
                all_rtgs.append(rtg_n)

                # Get Q-values
                if pruning_critic is not None:
                    qa_values_na = pruning_critic.qa_values(path['observation'])

                    qa_values_na = ptu.from_numpy(qa_values_na)
                    ac_n = ptu.from_numpy(path['action']).to(torch.long)
                    q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
                    q_values_n = list(ptu.to_numpy(q_values_n))
                    # print(q_values_n)
                    all_q_values.append(q_values_n)
                
                if pruned_policy is not None:
                    traj_actions = []
                    for ob in path['observation']:
                        ac_n_policy = pruned_policy.get_action(ob)
                        traj_actions.append(ac_n_policy)
                    policy_actions.append(traj_actions)

                if buffer_path is None:
                    buffer_path_list = []
                else:
                    buffer_path_list = buffer_path.split("_")

                    #If we have biomarkers we extract these as well and add them to the actions file
                    if 'biomarkers.pkl' in buffer_path_list:
                        # ['SOFA', 'SIRS', 'Arterial_lactate', 'Arterial_pH', 'BUN', 'HR', 'DiaBP', 'INR', 'MeanBP', 'RR', 'SpO2',
                        # 'SysBP', 'Temp_C', 'GCS', 'mechvent', 'paO2', 'paCO2']
                        sofa.append(path['SOFA'].tolist())
                        sirs.append(path['SIRS'].tolist())
                        art_lactate.append(path['Arterial_lactate'].tolist())
                        art_ph.append(path['Arterial_pH'].tolist())
                        bun.append(path['BUN'].tolist())
                        hr.append(path['HR'].tolist())
                        diabp.append(path['DiaBP'].tolist())
                        inr.append(path['INR'].tolist())
                        meanbp.append(path['MeanBP'].tolist())
                        rr.append(path['RR'].tolist())
                        spo2.append(path['SpO2'].tolist())
                        sysbp.append(path['SysBP'].tolist())
                        temp_c.append(path['Temp_C'].tolist())
                        gcs.append(path['GCS'].tolist())
                        mechvent.append(path['mechvent'].tolist())
                        pao2.append(path['paO2'].tolist())
                        paco2.append(path['paCO2'].tolist())

        # Log/save

        output_dict = {'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags, 'mortality_rtg': all_rtgs}

        if 'biomarkers.pkl' in buffer_path_list:
            output_dict.update({'SOFA': sofa, 'SIRS': sirs, 'Arterial_lactate': art_lactate, 'Arterial_pH': art_ph, 'BUN':bun, 'HR':hr, 'DiaBP':diabp, 
            'INR':inr, 'MeanBP':meanbp, 'RR': rr, 'SpO2': spo2, 'SysBP': sysbp, 'Temp_C': temp_c, 'GCS': gcs, 'mechvent': mechvent, 'paO2': pao2, 'paCO2': paco2})
        
        if pruning_critic is not None:
            output_dict.update({'q_vals': all_q_values})
        
        if pruned_policy is not None:
            output_dict.update({'policy_actions': policy_actions})
        
        with open(os.path.join(self.log_dir, 'actions.pkl'), 'wb') as f:
            pickle.dump(output_dict, f)
            
            """
            if pruning_critic is not None:
                if pruned_policy is not None:
                    pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                            'mortality_rtg': all_rtgs, 'q_vals': all_q_values, 'policy_actions': policy_actions}, f)
                else:
                    pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                                'mortality_rtg': all_rtgs, 'q_vals': all_q_values}, f)
            else:
                if pruned_policy is not None:
                    pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                                'mortality_rtg': all_rtgs, 'policy_actions': policy_actions}, f)
                else:
                    pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                                'mortality_rtg': all_rtgs}, f)
            """

        return opt_actions, pruned_actions

