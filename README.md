# anyrl

Package anyrl will implement various APIs for Reinforcement Learning. This will include training algorithms like [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), a gradient-based reinforcement learning algorithm. It will also include APIs to integrate with various RL environments.

# TODO

 * Way to record rollout batches from OpenAI Gym
 * Compute score function (grad of log-prob) from RolloutBatch
 * Compute Fisher-vector products
   * Make copy of model with anyfwd parameters
   * Compute KL from RolloutBatch
     * KL between constant Tape of outputs and variable outputs
 * Add option to do CG on subset of trajectories
   * lazyrnn Reduce() and potentially Prune() APIs.
