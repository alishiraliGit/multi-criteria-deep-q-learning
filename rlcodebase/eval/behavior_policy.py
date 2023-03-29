import os
import pickle

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from rlcodebase import configs
from rlcodebase.envs import mimic_utils
from rlcodebase.infrastructure.utils.rl_utils import convert_listofrollouts
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.dqn_utils import gather_by_actions
from rlcodebase.policies.base_policy import BasePolicy


class BehaviorPolicy(BasePolicy):
    def __init__(self):
        self.cls = LogisticRegression(multi_class='multinomial', penalty=None, max_iter=1000)

    def update(self, ob_no: np.ndarray, ac_n: np.ndarray, **kwargs) -> dict:
        self.cls.fit(ob_no, ac_n)

        return {'accuracy': self.cls.score(ob_no, ac_n)}

    def get_actions(self, ob_no: np.ndarray) -> np.ndarray:
        return self.cls.predict(ob_no)

    def get_action_probs(self, ob_no: np.ndarray, ac_n: np.ndarray) -> np.ndarray:
        probs_na = self.cls.predict_proba(ob_no)
        probs_n = ptu.to_numpy(gather_by_actions(ptu.from_numpy(probs_na), ptu.from_numpy(ac_n).to(torch.long)))
        return probs_n


if __name__ == '__main__':
    # ----- Settings -----
    settings = dict()

    # Env
    settings['env_name'] = 'MIMIC-Continuous'
    settings['env_rew_weights'] = [1, 0, 0, 0, 0, 0]

    # Offline
    settings['buffer_path'] = \
        os.path.join('Replay_buffer_extraction', 'Encoded_paths13_all_rewards_var1.pkl')

    # System
    settings['seed'] = 1

    # ----- Load and format data -----
    with open(settings['buffer_path'], 'rb') as f:
        all_paths = pickle.load(f)

    # Format rewards
    all_paths = mimic_utils.format_paths(all_paths, settings['env_name'], settings['env_rew_weights'])

    # Set aside the test data
    train_paths, test_paths = \
        train_test_split(all_paths, test_size=configs.EvalConfig.MIMIC_TEST_SIZE, random_state=settings['seed'])

    # ----- Get env params -----
    env_dims = mimic_utils.get_mimic_dims(train_paths)

    # ----- Prepare data for classification -----
    ob_tr_no, ac_tr_n, _, _, _, _ = convert_listofrollouts(train_paths)
    ob_te_no, ac_te_n, _, _, _, _ = convert_listofrollouts(test_paths)
    ac_tr_n = ac_tr_n.astype(int)
    ac_te_n = ac_te_n.astype(int)

    # ----- Classification with cross-entropy loss -----
    behavior_policy = BehaviorPolicy()

    print(behavior_policy.update(ob_tr_no, ac_tr_n))

    ac_pr_n = behavior_policy.get_actions(ob_te_no)
    probs_pr_n = behavior_policy.get_action_probs(ob_te_no, ac_te_n)
