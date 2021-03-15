---
title: "Part II - From AlphaGo to MuZero"
date: 2021-03-04T16:54:39+08:00
draft: true
author: SY Chou

description: "A paper review of Mastering the game of Go without human knowledge and Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm, as well as an introduction of AlphaGo Zero and AlphaZero"
categories: ["model-based RL", "RL"]
series: ["From Alphago to Muzero"]
tags: ["RL", "deep learning", "model-based RL"]

keywords:
- AlphaGo
- AlphaGo Zero
- AlphaZero
- MuZero
- model-based RL
- reinforcement learning
- RL
- deep reinforcement learning
- DRL
- deep learning
- DL
- DeepMind

cover:
    image: "img/just_imgs/alphago_zero.webp"
    relative: false
---

## Mastering the game of Go without human knowledge

The paper propose **AlphaGo Zero** which is known as self-playing without human knowledge.
### Reinforcement learning in AlphaGo Zero

![](/blog/img/alphago_to_muzero/alphago/alphago_zero_selfplay.png)

<!-- ![](/blog/img/alphago_to_muzero/alphago/alphago_zero_loss.png) -->

$$
(p, v) = f_{\theta}
$$

$$
l = (z - v)^2 - \pi^T log(p) + c||\theta||^2
$$

## Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

The paper propose **AlphaZero** which is known as self-playing to compete any kinds of board game.