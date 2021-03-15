---
title: "Part I - Toward NNGP and NTK"
date: 2021-03-15T23:34:57+08:00
draft: true

description: "An intuitive and friendly introduction of Neural Network Gaussian Process(NNGP)"
categories: ["deep learning", "machine learning", "NNGP", "NTK"]
series: ["Toward NNGP and NTK"]
tags: ["gaussian process", "deep learning", "machine learning", "bayes", "NNGP", "NTK"]

keywords:
- Gaussian process
- Gaussian
- gp
- neural network gaussian process
- NNGP
- neural tangent kernel
- NTK
- machine learning
- ml
- deep learning
- dl
- numerical optimization

cover:
    image: "img/just_imgs/flow_sea.jpg"
    relative: false
---

# Neural Network Gaussian Process(NNGP)
Model the neural network as GP, aka neural network Gaussian Process(NNGP). Intuitively, the kernel of NNGP compute the distance between the output vectors of 2 input data points.

We define the following functions as neural networks with fully-conntected layers:

$$z_{i}^{1}(x) = b_i^{1} + \sum_{j=1}^{N_1} \ W_{ij}^{1}x_j^1(x), \ \ x_{j}^{1}(x) = \phi(b_i^{0} + \sum_{k=1}^{d_{in}} \ W_{ik}^{0}x_k(x))$$

where $b_i^{1}$ is the $i$th-bias of the second layer(the same as first hidden layer), $W_{ij}^{1}$ is the $i$th-weights of the first layer(the same as input layer) , function $\phi$ is the activation function, and $x$ is the input data of the neural network. As a result,  

Thus, the kernel of $l$-th layer is

$$K_{NN}^l(x, x') = \sigma_b^2 + \sigma_w^2 E_{z_i^{l-1} \sim GP(0, K^{l-1})}[\phi(z_i^{l-1}(x)) \phi(z_i^{l-1}(x'))]$$