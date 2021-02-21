---
title: "Toward NNGP and NTK"
date: 2021-02-19T20:46:29+08:00
draft: true
author: SY Chou
# description: "Introduce"
categories: ["deep learning", "machine learning", "NNGP", "NTK"]
series: []
tags: ["gaussian process", "deep learning", "machine learning", "bayes", "NNGP", "NTK"]

cover:
    image: "img/just_imgs/wave_process.jpg"
    relative: false
---

# Neural Network Gaussian Process(NNGP)

# Neural Tangent Kernel(NTK)
If I have to summary NTK in a sentence, I would say
# *"NTK represent the changes of the weights before and after the gradient descent update"*

Let's start the journey of revealing the black-box neural networks.

## Setup a Neural Network

First of all, we need to define a simple neural network with 2 hidden layers

$$ y(x, w)$$

where $y$ is the neural network with weights $w \in \mathbb{R}^m$ and, $\\{ x, \bar{y} \\}$ is the dataset which is a set of the input data and the output data with $N$ data points. Since we focus on analyze the weight $w$, we simplify the notation $y(x, w)$ as $y(w)$

Suppose we have a regression task on the network $y(w)$, define

$$L(w) = \frac{1}{N} \frac{1}{2} \Vert y(w) - \bar{y} \Vert^2_2 $$

where $L(w)$ is our object loss which we want to minimize. Since the term $\frac{1}{N}$ is regardless to our goal, just ignore it and we get a simpler loss function

$$L(w) = \frac{1}{2} \Vert y(w) - \bar{y} \Vert^2_2 $$

## To The Limit of Infinite Width
In order to **measure the difference of the weights during training the neural network**, we define a normalized metric as following

$$\frac{\Vert w_n - w_0 \Vert_2}{\Vert w_0 \Vert_2}$$

where $w_n$ and $w_0$ are the weights at *n*-th training iteration and the initial weights. $\Vert w_n - w_0 \Vert_2$ means the quantity of the differnce between parameters $w_n$ and $w_0$ and it is normalized by the 2-norm $\Vert w_0 \Vert_2$

![losses with 3 widths](/img/nngp_ntk/losses_3widths.png)

![normalized weight-changes with 3 widths](/img/nngp_ntk/weightchange_3widths.png)

As we can see, **the difference of the weights during training decrease as the width of network grows**. As a result, **the trained weights should be very close to the inital weights $w_0$ as the width of network goes to infinity**.

## Apply Taylor Expansion
We've known the Taylor expansion is 

$$f(x) = \Sigma_{n=0}^{\infty} \ \frac{f^{n}(a)}{n!} (x - a)^{n}$$

A function $f(w)$ expanding on the $w_0$ with first order approximation is 

$$f(w) \approx f(w_0) + \frac{df(w_0)}{dw} (w - w_0)$$

It is trivial that if $w$ is a vector, we need to replace the derivative $\frac{df(w_0)}{dw}$ with gradient $\nabla_{w} f(w_0)^{\top}$

$$f(w) \approx f(w_0) + \nabla_{w} f(w_0)^{\top} \ (w - w_0)$$

Apply to the network $y(w)$

$$y(w) \approx y(w_0) + \nabla_{w} y(w_0)^{\top} \ (w - w_0)$$

However, the most difficult thing is **how can we guarantee the bound is correct?** It is so complex that I wouldn't put it in this article but I will provide an intuitive explaination of **what does NTK mean?** in the following article. Please keep reading it if you are interested in it.

## Flow And Velocity Field
Before diving into NTK more deeply, we need to understand what is **Flow** and **Velocity Field**.

### Velocity Field

### Flow

## Gradient Flow
We've know the update of the gradient descent is

$$w_{t+1} = w_t - \eta \nabla_{w} L(w_t)$$

Let the function $w(t) = w_t$ and define the gradient flow over weights is $\dot{w}(t)$

$$\dot{w}(t) = - \nabla_{w} L(w(t))$$

Actually, the meaning of the gradient flow $\dot{w}(t)$ is likey **the changing direction of gradient descent**.

We expand the gradient of the loss function with chain rule

$$
\dot{w}(t) = - \nabla_{w} L(w(t)) = - \nabla_{w} \frac{1}{2} \Vert y(w(t)) - \bar{y} \Vert^2_2
$$

$$
= - \frac{1}{2} \cdot 2 \nabla_{w} y(w(t)) (y(w(t)) - \bar{y}) = - \nabla_{w} y(w) (y(w(t)) - \bar{y})
$$

Now we can derive the flow of the network $\dot{y}(w(t))$

$$
\dot{y}(w(t)) = \nabla_{w} y(w(t))^{\top} \dot{w}(t) 
$$

$$
= -\nabla_{w} y(w(t))^{\top} \nabla_{w} y(w) (y(w(t)) - \bar{y}) = - \nabla_{w} y(w_t)^{\top} \nabla_{w} y(w_t)(y(w_t) - \bar{y})
$$

Actually, we are now very close to the neural tangent kernel(NTK). The NTK is a kernel matrix defined as

## $$\Sigma_{NTK}(w_t, w_t) = \nabla_{w} y(w_t)^{\top} \nabla_{w} y(w_t)$$

Since the weights of the infinite-width network doesn't change during the training. 

$$y(w_t) \approx y(w_0)$$

We get

$$
-\nabla_{w} y(w_t)^{\top} \nabla_{w} y(w_t) \approx -\nabla_{w} y(w_0)^{\top} \nabla_{w} y(w_0)
$$

## $$= \Sigma_{NTK}(w_0, w_0)$$

Again, $\Sigma_{NTK}(w_0, w_0)$ is the Neural Tangent Kernel, NTK.

It is very surprise that **NTK doesn't depend on the input data but the inital weights**. Well, why doesn't NTK depend on the input data? Actually, it is proved by [another work](https://arxiv.org/abs/2012.00152) that neural network is just a kernel machine. It is a quite interesting work but I wouldn't cover in this article. 

To summary, **the weights of an infinite width network almost don't change during training. As a result, the kernel always stay almost the same.** We can use NTK to analyze many properties of neural network and the neural networks are no longer black boxes.
## Another Explaination Without Flow

# Papers
NNGP
- [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)

NTK
- [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720)

# Reference
Gaussian Distribution
- [StackExchange - Product of two multivariate normal distribution](https://math.stackexchange.com/questions/3495719/product-of-two-multivariate-normal-distribution)

NNGP
- [Deep Gaussian Processes](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html)

NTK
- [Understanding the Neural Tangent Kernel By Rajat's Blog](https://rajatvd.github.io/NTK/)
  - Code for the blog [rajatvd/NTK](https://github.com/rajatvd/NTK)
- [Ultra-Wide Deep Nets and the Neural Tangent Kernel (NTK)](https://blog.ml.cmu.edu/2019/10/03/ultra-wide-deep-nets-and-the-neural-tangent-kernel-ntk/)
- [CMU ML Blog: Ultra-Wide Deep Nets and the Neural Tangent Kernel (NTK)](https://blog.ml.cmu.edu/2019/10/03/ultra-wide-deep-nets-and-the-neural-tangent-kernel-ntk/)
- [Some Intuition on the Neural Tangent Kernel](https://www.inference.vc/neural-tangent-kernels-some-intuition-for-kernel-gradient-descent/)
- [直观理解Neural Tangent Kernel](https://zhuanlan.zhihu.com/p/339971642)

Flow
- [Let it flow - Gradient flow and gradient descent](http://awibisono.github.io/2016/06/13/gradient-flow-gradient-descent.html)

- [Max Planck Science - Gradient Flow I](https://www.youtube.com/watch?v=pesXn-qwMvQ)
- [StackExchange - gradient flow and what is, for example, L2 gradient?](https://math.stackexchange.com/questions/156236/gradient-flow-and-what-is-for-example-l2-gradient)

Notebook
- [Colab Notebook](https://colab.research.google.com/drive/1tLCfu0DCqN3RxLHA9rIARjBtC1VOHFNg?usp=sharing)