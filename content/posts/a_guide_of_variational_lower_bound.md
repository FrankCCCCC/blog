---
title: "A Guide Of Variational Lower Bound"
date: 2021-02-23T12:39:16+08:00
draft: true

categories: ["statistics"]
series: []
tags: ["statistics", "information theory"]

cover:
    image: "img/just_imgs/whirl_flow.jpg"
    relative: false
---

## Problem Setup

It is also knowd as **Evidence Lower Bound(ELBO)** or **VLB**. We can  assume that $X$ are observations (data) and $Z$ are hidden/latent variables. In general, we can also imagine $Z$ as a parameter and the relationship between $Z$ and $X$ are represented as the following

![](/blog/img/a_guide_of_variational_lower_bound/elbo_graph.png)

In the mean time, by the definition of Bayes' Theorem and conditional probability, we can get

$$p(Z | X) = \frac{p(X | Z) p(Z)}{p(X)} = \frac{p(X | Z) p(Z)}{\int_{Z} p(X, Z)}$$

<!-- ![](/blog/img/a_guide_of_variational_lower_bound/elbo_bayes.png) -->


## Jensen's Inequality

It states that for the convex transformation $f$, the mean $f(w \cdot x + (1 - w) y)$ of $x$, $y$ on convex transform $f$ is less than or equal to the mean applied after convex transformation $w \cdot f(x) + (1 - w) f(y)$.

![](/blog/img/a_guide_of_variational_lower_bound/jensen.png)

Formaly, corresponding to the notation of the above figure, the Jensen's inequality can be defined as

<!-- ![](/blog/img/a_guide_of_variational_lower_bound/jensen_formular.svg) -->

$$f(t x_1 + (1 - t) x_2) \leq t f(x_1) + (1 - t) f(x_2)$$

In probability theory, for a random variable $X$ and a convex function $\varphi$, we can state the inequality as 

<!-- ![](/blog/img/a_guide_of_variational_lower_bound/jensen_prob.svg) -->

$$\varphi \ (E[X]) \leq E[\varphi(X)]$$