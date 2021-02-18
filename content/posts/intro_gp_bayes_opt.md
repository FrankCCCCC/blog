---
title: "A Very Brief Introduction to Gaussian Process and Bayesian Optimization"
date: 2021-02-16T17:28:58+08:00
draft: False
author: SY Chou

cover:
    image: "img/just_imgs/ny_skyline.jpg"
    relative: false # To use relative path for cover image, used in hugo Page-bundles
---

# Gaussian Process

## Big Picture and Background

Intuitively, Gaussian distribution define the **state space**, while Gaussian Process define the **function space**


Before we introduce Gaussian process, we should understand Gaussian distriution at first. For a RV(random variable) $X$ that follow Gaussian Distribution $\mathcal{N}(0, 1)$ should be following image:

![](/img/gp/normal01.png)

The P.D.F should be

$$x \sim \mathcal{N}(\mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{1}{2} (\frac{- \mu}{\sigma})^2}$$

As for Multivariate Gaussian Distribution, given 2 RV $x$, $y$ both 2 RV follow Gaussian Distribution $\mathcal{N}(0, 1)$ we can illustrate it as

![](/img/gp/multivariate_gaussian.png)

The P.D.F should be

For a set of random variables $X = (x_1, ..., x_k)$ that follow Gaussian distribution

$$(x_1, ..., x_k) \sim \mathcal{N}(\mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} e^{- \frac{1}{2} (X- \mu)^{\top} \Sigma^{-1} (X- \mu)}$$

where $\mu$ is the mean and $\Sigma$ is the covariance matrix.

The Gaussian process can be regarded as a **function space**, for example, given a function $f(x)$ The Gaussian process can be illustrated as following:

![](/img/gp/gp.png)

The blue solid line represent the mean of the Gaussian process and the shaded blue area represent the standard deviation(which means the uncertainty of the RV) for the corresponding RV. For example, while $x=-4$, the function $f(4) = \mathcal{N}(0, 2)$. That means the Gaussian process gives a Gaussian distribution $\mathcal{N}(0, 2)$ to describe the possible value of $f(-4)$. The most likely value of $f(-4)$ is 0 (which is the mean of the distribution). As the figure shows, the Gaussian process is quite simple that the mean function is a constant 0 and the standard deviation is 2.

The dotted line are the functions sampled from the Gaussian process. Each line gives a mapping function from $x$ to $f(x)$.

Note that the explaination above is from the point of view of function approximation. From the perspective of random process, the Gaussian process can be regarded as a time-variant system that the distribution is changing along the time.

![](/img/gp/gp_posterior.png)
![](/img/gp/gp_3d.png)

## Definition

A Gaussian process is a time continuous stochastic process ${x_t; t \in T}$ is Gaussian if and only if for every finite set of indices $t_1, ..., t_k$ in the index set $T$, $x_{t1},...x_{tk} = (x_{t1}, ..., x_{tk})$ is a multivariate Gaussian random variable.

For example, any point $x_1, ... x_N \in X, X \in \mathbb{R}^d$(Real Number with dimension $d$) is assigned a random variable $f(x)$ and where the joint distribution of a finite number of these variables $p(f(x_1),…,f(x_N))$ is itself Gaussian:

<!-- ![](/img/gp/gp_def.png) -->

$$p(f|X) = \mathcal{N}(f|\mu, K)$$ 

where $\mu$ is a vector which consists of **mean function** and $K$ is a covariance matrix which consists of **covariance function** or **kernel function $\kappa$**. The set of mean function $\mu = (m(x_1),…,m(x_N))$ give the mean value over set $X$. The set of kernel function is $K={K_{ij} = \kappa(x_i,x_j)) where x_i, x_j \in X}$ which define the correlaton between 2 values $x_i$ and $x_j$. 

Note that a data point $x_i$ or $x_j$ might be multi-dimensions. **The kernel functions may defined on the vectors as well.**

## Kernel

To understand the kernel function intuitively, the kernel function can be regarded as a kind of **distance metric** which give the distance in another space. For example, the kernel $k(x_i, x_j) = {x_i}^2 + {x_j}^2$ map the Cartesian coordinate to polar coordinate and convert the Euclidean distance into radius. 

Some common kernels are:
- Constant Kernel:
  
  $K_C(x_i, x_j) = C$
  <!-- ![](/img/gp/const_kernel.svg) -->

- RBF Kernel:
  
  $K_{RBF}(x_i, x_j) = e^{-\frac{|| x_i - x_j ||^2}{2 \sigma^2}}$
  <!-- ![](/img/gp/rbf_kernel.svg) -->

- Periodic Kernel
  
  Suitable for periodic relation

  $K_{P}(x_i, x_j) = e^{-\frac{2 \sin^2 (\frac{d}{2})}{\ell^2}}$
  <!-- ![](/img/gp/periodic_kernel.svg) -->

- Polynomial Kernel
  
  $K_{Poly}(x_i, x_j) = ( x_i^{\top}x_j+ c)^d$
  <!-- ![](/img/gp/polynomial_kernel.svg) -->

- Neural Network Kernel
  
  Model the neural network as GP, aka neural network Gaussian Process(NNGP)
  For more detail, please refer to [this slide](https://www.brown.edu/research/projects/crunch/sites/brown.edu.research.projects.crunch/files/uploads/Pang-NNGP-Crunch-Seminar.pdf).

For more detail, please refer to [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/). You can play with interactive widgets on the website.

## Inference

Given a training dataset with noise-free function values $f$ at inputs $X$, a GP prior can be converted into a GP posterior $p(f^{\ast}|X^{\ast},X,f)$ which can then be used to make predictions $f^{\ast}$ at new inputs $X^{\ast}$ By definition of a GP, the joint distribution of observed values $f$ and predictions $f^{\ast}$ is again a Gaussian which can be partitioned into

<!-- ![](/img/gp/gp_join.png) -->

$$
\begin{pmatrix}
f\newline
f^{\ast}
\end{pmatrix}
\sim \mathcal{N}
\begin{pmatrix}
0, 
\begin{pmatrix}
K & K^{\ast}\newline
K^{\ast \top} & K^{\ast \ast}
\end{pmatrix}
\end{pmatrix}$$

where $K^{\ast} = \kappa(X,X^{\ast})$ and $K^{\ast \ast} = \kappa(X^{\ast},X^{\ast})$. With $N$ training data and $N^{\ast}$ new input data $K$ is a $N×N$ matrix , $K^{\ast}$ a $N×N^{\ast}$ matrix and $K^{\ast \ast}$ a $N^{\ast}×N^{\ast}$ matrix. 

We've known conditional distribution rules.


As a result, the predictive Gaussian Distribution is 

$$p(f^{\ast} | X^{\ast},X,f) = \mathcal{N}(f^{\ast} | \mu^{\ast}, \Sigma^{\ast})$$

$$\mu^{\ast} = K^{\ast \top} K^{-1} f$$

$$\Sigma^{\ast} = K^{\ast \ast} - K^{\ast \top} K^{-1} K^{\ast}$$

However, the above equations don't consider the effect of noise. Suppose we need to evalutate a noisy model $y=f+\epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma_y^2 I)$ is the noise. The noise follows normal distribution and has a covariance matrix $\sigma_y^2 I$. Thus, the predictive distribution is 

$$p(f^{\ast} | X^{\ast},X,y) = \mathcal{N}(f^{\ast} | \mu^{\ast}, \Sigma^{\ast})$$

$$\mu^{\ast} = K^{\ast \top} K_{y}^{-1} y$$

$$\Sigma^{\ast} = K^{\ast \ast} - K^{\ast \top} K_{y}^{-1} K^{\ast}$$

where $K_{y}^{-1} = K + \sigma_y^2 I$(linearilty of Gaussian distribution). Finally, we also want to replace the noise-free prediction $f^{\ast}$ with noisy prediction $y^{\ast}$. We can derive

$$p(y^{\ast} | X^{\ast},X,y) = \mathcal{N}(y^{\ast} | \mu^{\ast}, \Sigma^{\ast} + \sigma_y^2 I)$$

Finally, we get the probability of noisy prediction $y^{\ast}$ which conditions on noisy training dataset $X,y$ and test dataset $X^{\ast}$.

---
# Bayesian Optimization

In many machine learning or optimization problem, we need to optimize an unkown object function $f$. One of the solutions to optimize function $f$ is **Bayesian Optimization**. Bayesian Optimization assume the object function $f$ follows a distribution or prior model. This prior model is called **surrogate model**. We sample from the object function $f$ and approximate the function $f$ with surrogate model. The extra information like uncertainty provided from surrogate model contribute to the sample-efficiency of Bayesian optimization. In the mean time, we also use the **acquisition function** to choose the next sampling point.

## Definition
Formaly, suppose we have a block-box function $f : X \to R$ that we with to minimize on some domain $x \subseteq X$ . That is, the Bayesian optimization wish to find

### `$\begin{equation} x^{\ast} = \mathop{\arg\max}_{x \in X} \ \ f(x) \end{equation}$`

If we use Gaussian Process as prior(Surrogate Model), we can get

$p(f) = GP(f; µ, K).$

Given observations $D = (x,f)$, we can condition our distribution on $D$ as usual

$p(f | D) = GP(f; µ_{f|D}, K_{f|D})$

## Surrogate Model
A popular model is Gaussian Process. Gaussian process defines a prior over functions and provides a flexiable, powerful and, smooth model which is especially suitable for dynamic models.

## Algorithm
The Bayesian optimization procedure is as follows. 

---

For index $t=1,2,…$ and an acquisition function $a(x|D)$

repeat:
- Find the next sampling point $x_t$ by optimizing the acquisition function over the surrogate model: $x_t=argmax_{x \in X} \ a(x|D_{1:t−1})$
- Obtain a possibly noisy sample $y_t=f(x_t)+\epsilon_t$ from the objective function $f$.
- Add the sample to previous samples $D_{1:t}=D_{1:t−1},(x_t,y_t)$ and update the surrogate model.

---

## Probability improvement(PI) Method
A naive idea is always evaluating the points with lowest value. The PI method do the same thing exactly.
However, the Gaussian Process gives a distribution of $f(x_i)$ on point $x_i$. As a result, in practice, we give an threshold $f'$, integrate the probability of $x_i < f'$ and, pick the point with highest probability.

Formaly, we can define an utility function $u(x)$

$$u(x) = \begin{cases}
  0, \ \ f(x) > f'\newline
  1, \ \ f(x) \leq f'
\end{cases}$$

Then, integrate the probability of $f(x) \leq f'$

$$
a_{PI}(x|D) = \mathbb{E}[u(x) | x, D] = \int_{-\infty}^{f'} \mathcal{N}(f; \mu(x), \kappa(x, x)) \ df
=\phi(f'; \mu(x), \kappa(x, x))
$$

Finally, we choose next evaluating point $x_t$ with highest probability $a_{PI}(x|D_{1:t−1})$

### `$\begin{equation} x_t = \mathop{\arg\max}_{x \in X} \ \ a_{PI}(x|D_{1:t−1}) \end{equation}$`

## Expected improvement(EI) Method
PI algorithm is easy to understand. However, **PI might get stuck in local optimal and underexplore globally**. A better way is that we evaluates $f$ at the point that, in expectation, improves upon $f'$ the most. That is the idea of EI algorithm.

Formaly, we suppose that $f'$ is the minimal value of $f$ observed so far. We can define an utility function as following:

$$u(x) = \mathop{\max} (0, f' − f(x))$$

The ultility function can be regarded as the advanrage versus the average. Then, we can derive the expectation via integration

$$
a_{EI}(x|D) = \mathbb{E}[u(x) | x, D] = \int_{-\infty}^{f'} (f' - f) \mathcal{N}(f; \mu(x), \kappa(x, x)) \ df
$$

$$
=(f' - \mu(x))\phi(f'; \mu(x), \kappa(x, x)) + \kappa(x, x) \mathcal{N}(f'; \mu(x), \kappa(x, x))
$$

We always choose the next evaluating point $x_t$ which has highest $a_{EI}(x|D_{1:t−1})$, thus

### `$\begin{equation} x_t = \mathop{\arg\max}_{x \in X} \ \ a_{EI}(x|D_{1:t−1}) \end{equation}$`

Intuitively, the term $(f' - \mu(x))\phi(f'; \mu(x), \kappa(x, x))$ can be taken as exploitation(it encourage to evaluate the point with higher reward, lower $\mu(x)$), since it means how much advantage does point $x$ has? The term $\kappa(x, x) \mathcal{N}(f'; \mu(x), \kappa(x, x))$ represents how much uncertainty does point $x$ has?, so it can be viewed as exploration(it encourage to evaluate the point with higher uncertainty, higher $\kappa(x, x)$). **EI algorithm can trade off the exploration and exploitation automatically** and also the most popular algorithm of Bayesian Optimization.

## Bayesian Upper Confident Bound(UCB) Method
Before diving to Bayesian UCB method, please understand the bandit problem first. 

Bayesian UCB inherents UCB. They both give a relation between upper bound and probability confidence. The different thing is UCB finds the relation with Hoeffding's Inequality while Bayesian UCB find the relation with Gaussian distribution itself. 

For example, it is common that we know if we sample values from Gaussian distribution, 95% of them are between the mean plus 2 standard deviation and mean subtract 2 standard deviation.

![](/img/gp/gaussian_dist_conf.png)

## Entropy Search

# Reference 

- [Bayesian Optimization & Algorithm Configuration](https://www.youtube.com/watch?v=6D9Rqda0dpg&feature=youtu.be)

  Bayesian optimization of CS159

- [Machine Learning by mathematicalmonk](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA)
  
  Give an intuitive explaination of the math often used in ML.

- [(ML 19.1) Gaussian processes - definition and first examples](https://www.youtube.com/watch?v=vU6AiEYED9E&list=PLD0F06AA0D2E8FFBA&index=150)

  By mathematicalmonk.

- [UAI 2018 2. Bayesian Optimization](https://www.youtube.com/watch?v=C5nqEHpdyoE)
  
  A complete and clear talk for the beginner of Bayesian Optimization.
- [Gaussian processes by Martin Krasser](http://krasserm.github.io/2018/03/19/gaussian-processes/#References)
  
  The blog provide an excellent and intuitive explanation of Gaussian Process with Python example code.

- [Bayesian optimization by Martin Krasser](http://krasserm.github.io/2018/03/21/bayesian-optimization/)
  
  The blog provide an excellent and intuitive explanation of Bayesian Optimization with Python example code.

- [Coursera: Bayesian optimization](https://www.coursera.org/lecture/bayesian-methods-in-machine-learning/bayesian-optimization-iRLaF)
  
  The short video provide a very high-level explaination of Bayesian Optimization. Recommend for beginner.

- [Gaussian Processes - Part 1](https://www.youtube.com/watch?v=OdCXdUzLfao)

  The video provide a detailed explaination of Gaussian Process.

- [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/?fbclid=IwAR3XSg_gQ9KvIG9qPOXCWjGGEhl7b3qSZCLxXeee-uDbuQtktLCf-2lVeno#DimensionSwap)

   Provide a lot of interactive widgets to play around with Gausssian Process but fewer math.

- [StackExchange: Intuitive Understanding of Expected Improvement for Gaussian Process](https://stats.stackexchange.com/questions/426782/intuitive-understanding-of-expected-improvement-for-gaussian-process)
- [ML Tutorial: Gaussian Processes (Richard Turner)](https://www.youtube.com/watch?v=92-98SYOd)
- [10 Gaussian Processes, pt 1/3 Basics](https://www.youtube.com/watch?v=AEf_ta4vyKU)


- [Bzarg: How a Kalman filter works, in pictures](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
  
  Intuitively explain Kalman Filter with picture & examples.
  
- [图说卡尔曼滤波，一份通俗易懂的教程](https://zhuanlan.zhihu.com/p/39912633)

  Intuitively explain Kalman Filter with picture & examples. The article is translated from Bzarg: How a Kalman filter works, in pictures.

- [The Kalman Filter [Control Bootcamp]](https://www.youtube.com/watch?v=s_9InuQAx-g)

  The video provide the mathmatical proof for Kalman Filter.
- [(PP 6.9) Conditional distributions of a Gaussian](https://www.youtube.com/watch?v=G6_OdMXpiVY)
- [Deriving the conditional distributions of a multivariate normal distribution](https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution)