# About this example

In this example, we train a neural network on the CartPole-v0 environment in OpenAI Gym. The network typically learns after several hundred episodes.

Overall, this demo is not extremely data efficient. It uses large batches and takes at least 30 batches to start converging. An actor-critic solution or a model-based method would likely work much better.

For the results of this demo, see [this page on gym.openai.com](https://gym.openai.com/evaluations/eval_jCncBhVQNmFH4tUc3sGTw).
