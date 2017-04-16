# trpo

Package trpo will implement [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), a gradient-based reinforcement learning algorithm.

# TODO

 * JoinRolloutBatches
   * Make lazyrnn Join API
 * Way to record rollout batches from OpenAI Gym
   * Make lazyrnn Writer API - create Seq dynamically
   * Make lazyrnn MemRereader API - create Rereader from Seq
     * Potentially create DiskRereader too
 * Compute score function (grad of log-prob) from RolloutBatch
   * Make lazyrnn Map and MapN APIs
 * Compute KL from RolloutBatch
   * Make lazyrnn Mean aggregate
