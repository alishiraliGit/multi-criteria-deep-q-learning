# Overview
This file summarizes key results of my Health RL litereature review. Key focus will be around: Problem formulation? Use of MIMIC data? Set-up of toy problems? Existence of semi-synthetic experiments?

## Idea -- What could be the focus of this paper

A. Show that pruning as pre-processing enables more effective learning of policies (Perspective of CS285 write-up)
[Application: An improved RL-based Sepsis detection model]
--> Evaluation could be inspired by Komorowski et al. (2018) to test whether th learned policy is better

B. Use pruning to flag "bad" actions? 
[Application: A physician alert system, potentially allowing them to avoid suboptimal treatment decisions]
--> Evaluation could be inspired by Fatemi et al. (2021); show that pruned state-action pairs are associated with increased mortality. Analyze Q-values of trajectories of patients after "bad" action (potential split by survivors and non-survivors)
--> Here the focus would be primarily on the pruning algorithm, since the idea is to build an algorithm that identifies (and thus allows to avoid) bad actions
--> Potentially we could implement an evaluation on a Toy problem similar to LifeGate in Fatemi et al. (2021) where our pareto pruning step should be able to discard all the actions that go into the bad region.

## Idea -- Hack to get less sparse death reward

In our last call we had the question about how one could transform the sparse end-stage mortality reward into a intermediate measure. Here is one crude idea:

0. Ensure states are discretized
1. For each state count how often it was part of a transition that lead to survival vs. that lead to death
2. Calculate survival transition probability (survival_transition_count / (survival_transition_count + death_transition_count))
3. Calculate reward: survival_reward * survival transition probability

Do this procedure for each state. Potentially exclude states that appear less than X (e.g. 5 times).

## Some technical ideas

1. Consider aligning processing of MIMIC data with Fatemi et al. (2021) repo (the MIMIC sepsis processing repo of Microsoft Research)

## Fatemi et al. (2021) Medical Dead-ends and Learning to Identify High-risk States and Treatments
Repo of the paper: https://github.com/microsoft/med-deadend
Repo for processing: https://github.com/microsoft/mimic_sepsis

### Motivation

Offline RL has many weaknesses. Hence, reframe the problem and apply RL instead to identify dead-end states i.e. states from which no action can be made to achieve positive end stage reward. 

Idea: Identify behaviours to avoid and thus constrain the space of policies --> **This is very similar to what we are trying to do with Pareto Learning**

### Problem formulation

Policy condition: If selecting an action in a state leads to a dead-end with probability of e.g. 80%, then this action should be selected by a policy at maximum 20% of the time.

Have a separate MDP for dead-ends and rescue states. Deadend MDP rewards -1 for transition to deadend or negative terminal state.
Rescue MDP rewards 1 for transition to deadend or negative terminal state.

Authors show that policy condition can be implemented by pi(s,a) < 1+ Qd(s,a), this is used to impelement dead-end identification

(see theoreme 1 on page 5 for more details)

### Experimental set-up

1. Toy set-up LifeGate
Very simple tabular environment where part of the environment are deadends that always ultimately lead to a negative terminal state. Apply training procedure and show that the pre-specified dead-ends indeed become flagged as dead-end states by the Value functions for deadend discorvery and have a value of 0 assigned by the value functions for the rescue state discovery. The algorithm requires a threshold parameter that determines how easily states get flagged as deadends. (this may be similar to our Pareto stength parameter)

2. MIMIC analysis
Use MIMIC III
Use state construction network to define states (code is here: https://github.com/microsoft/mimic_sepsis.)
Use double-DQN to train each network (deadend and rescue)
All models are trained on 75% of cohort validated on 5% and final reported result from remaining 20% hold-out set.

Mititgation of data-imbalance:
Finally, to mitigate the data imbalance between surviving and non-surviving patients we use an additional data buffer that contains only the last transition of nonsurvivors trajectories. Thus, a stratified minibatch of size 64 is constructed of 62 samples from the main data, augmented with 2 samples from this additional buffer, all selected uniformly. This same minibatch structure is used for training each of the three networks.

Define different threshold values for red (stricter threshold) and yellow flags. The values choosen are tuned to minimize false positives and false negatives. 

Analysis conducted:
- Flag duration for ICU patients (i.e. when do patients remain in flagged states, want to show that non-surving patients remain for longer in flagged states)
- Histogram of Median value of V and Q network for surviving and non-surviving patients (want to show that non-surving patients average values become worse as time in the ICU increases while they stay similar for survivors. Hence this shows that V and Q values correlate with mortality)
- Table share of flagged survivors by time in the ICU for survivors and non-survivors (shows that after longer time in ICU share of flagged non-survivors increases while share of survivors stays the same, again indicated relationship between Q-values and mortality)
- Trend in vital signs and Q values before and after first flag for survivors and non-survivors

First flag analaysis (analyze moment when first flag was raised for a patient)
- Show Q and V values of survivors and non-survivors diverge after first flag, after flag these values drop
- Show vitals after first flag, after flag they become more severe
- Show Q and V values along a patients trajectory (observe jump 4h before diagnosed sepsis onset)

- Show Q-value development for survivors and non-surviviors

Patient trajectory analysis:
- Show that clinical notes indicate dead-end 
- Many vitals follow Q-values

### Overall take

A lot of the analyses here focus on the value functions which their approach enables. In our scenario the key output is a list of dominated state action pairs. In that spirit, our work should further analyze these state-action pairs and for instance show whether they are indeed associated with worse outcomes when when they appear in the dataset. 

Some ideas could be: development in vital signs after bad state-action pairs, survival rate compared to other actions taken in that state, look at trajectories and potentially mutliple occurences of weak state-action pairs. 

## Komorowski et al. (2018) AI Clinician

--> Evaluation conducted here could serve as inspiration for to evaluate the model that we built on top of the pruned action space 
--> Also it might be good to understand the performance reported in the paper better to compare it to our results, allowing us to test the validity of our implementation.

### Problem fomulation
Leverage policy-iteration to derive vasopressor and intravenous fluid recommendations for septic patients in the ICU.

### Evaluation

- Off-policy evaluation using weighted importance sampling, (something questionable: authors trained 500 different models using 500 different types of state clusters and showed that best model lower bound with these consistently surpassed 95% confidence upper bound of the AI policy)
- Plot action return against mortality risk. However not as correlation coefficient but with mean mortality for different action return brackets.
- Plot distribution of average policy return of survivors and non-survivors
- plotted distribution of policy value (likely estimated using importance sampling)
- Using bootstrapping with 2,000 resamplings, the median value of clinicians' policy and the AI policy were estimated at 56.9 (interquartile range, 54.7-58.8) and 84.5 (interquartile range, 84.3-87.7), respectively.
- Plot recommended dosage vs. given
- Compare return when AI policy and clinician decision match to the rest
- Mortality in case of variation from AI policy

### Deep-dive Importance sampling evaluation (page 7 of the paper)

- Idea use data generated under pi_0 (clinician policy) to evaluate AI policy (pi_1)
- Weighted importance sampling corrects for the discrepancy between pi_0 and pi_1
- Make AI policy stochastic (take 99% of time AI action and 1% random action) so WID can be applied across all trajectories 
- See formulas to apply importance weight (ration of policy suggestion of pi_o and pi_1) to evaluate trajectories
