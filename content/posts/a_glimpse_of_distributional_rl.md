---
title: "A Glimpse of Distributional RL"
date: 2021-02-16T20:36:18+08:00
draft: true

cover:
    image: "img/just_imgs/distributional_rl_prism.webp"
    relative: false
---

# Introduction
In traditional reinforcement learning, an agent predict a value for the state-action pair. The distributional RL predicts a **distribution of value** for the pair.

The advantages of distributional RL is that the agent can improve the estimation with more information and quickly. In the mean time, the agent can be sensitive to the risk of the action. It's very useful for some application like safe reinforcement learning, self-driving car etc...

# Wasserstein Metric

# Contraction & Bellman Optimality On Distribution