# Overview
This file summarizes key results of my Health RL litereature review. Key focus will be around: Problem formulation? Use of MIMIC data? Set-up of toy problems? Existence of semi-synthetic experiments?

## Random idea

In our last call we had the question about how one could transform the sparse end-stage mortality reward into a intermediate measure. Here is one crude idea:

0. Ensure states are discretized
1. For each state count how often it was part of a transition that lead to survival vs. that lead to death
2. Calculate survival transition probability (survival_transition_count / (survival_transition_count + death_transition_count))
3. Calculate reward: survival_reward * survival transition probability

Do this procedure for each state. Potentially exclude states that appear less than X (e.g. 5 times).

## Fatemi et al. (2021) Medical Dead-ends and Learning to Identify High-risk States and Treatments

