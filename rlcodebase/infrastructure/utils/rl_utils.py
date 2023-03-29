import numpy as np
import copy


#####################################
# General operations
#####################################

def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):
    data = copy.deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data


def discounted_cumsum(rewards: np.ndarray, gamma) -> np.ndarray:
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns a list where the entry in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """
    n = len(rewards)
    if not isinstance(gamma, np.ndarray):
        gamma = np.array(gamma)

    reversed_discounted_rewards = ((gamma ** range(n)) * rewards)[::-1]
    cumsum_rewards = np.cumsum(reversed_discounted_rewards)[::-1]
    reward_to_go_norm_coeff = (1 / gamma) ** range(n)
    rewards_to_go = cumsum_rewards * reward_to_go_norm_coeff

    return rewards_to_go


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


#####################################
# On-policy sampling
#####################################

def sample_trajectory(env, policy, max_path_length, render=False):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)  # HINT: query the policy's get_action function
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = 0  # HINT: this is either 0 or 1
        if done or steps >= max_path_length:
            rollout_done = 1
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
        Collect rollouts using policy
        until we have collected min_timesteps_per_batch steps
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """
        Collect ntraj rollouts using policy
    """
    paths = []

    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

    return paths


#####################################
# Path operations
#####################################


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs:  # checks image_obs != []
        image_obs = np.stack(image_obs, axis=0)

    return {
        'observation': np.array(obs, dtype=np.float32),
        'image_obs': np.array(image_obs, dtype=np.uint8),
        'reward': np.array(rewards, dtype=np.float32),
        'action': np.array(acs, dtype=np.float32),
        'next_observation': np.array(next_obs, dtype=np.float32),
        'terminal': np.array(terminals, dtype=np.float32)
    }


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path['observation'] for path in paths])
    actions = np.concatenate([path['action'] for path in paths])
    next_observations = np.concatenate([path['next_observation'] for path in paths])
    terminals = np.concatenate([path['terminal'] for path in paths])
    concatenated_rewards = np.concatenate([path['reward'] for path in paths])
    unconcatenated_rewards = [path['reward'] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


def get_pathlength(path):
    return len(path['reward'])


#####################################
# Schedules
#####################################

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, _t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meaning that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        indices = [e[0] for e in endpoints]
        assert indices == sorted(indices)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
