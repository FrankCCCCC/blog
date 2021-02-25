---
title: "Simple Guide Of VDN And QMIX"
date: 2021-02-26T01:13:41+08:00
draft: true
---

## Value-Decomposition Network(VDN)

## QMIX

The QMIX imporve the VDN algorithm via give a more general form of the contraint. It defines the contraint like 

$$\frac{\partial Q_{tot}}{\partial Q_{a}} \geq 0, \forall a$$

where $Q_{tot}$ is the joint value function and $Q_{a}$ is the value function for each agent.

An intuitive eplaination is that we want the weights of any individual value function $Q_{a}$ are positive. If the weights of individual value function $Q_{a}$ are negative, it will discourage the agent to cooperate, since the higher $Q_{a}$, the lower joint value $Q_{tot}$. Trivially, the contraint of VDN is just a special case that $\frac{\partial Q_{tot}}{\partial Q_{a}} = 1$.

# Papers

IQL

[Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1702.08887)

VDN

[Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)

QMIX

[QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)



# Reference
- [多智能体强化学习算法 QMIX：Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://mayi1996.top/2020/08/07/QMIX-Monotonic-Value-Function-Factorisation-for-Deep-Multi-Agent-Reinforcement-Learning/)
- [BAIR Blog - Scaling Multi-Agent Reinforcement Learning](https://bair.berkeley.edu/blog/2018/12/12/rllib/)