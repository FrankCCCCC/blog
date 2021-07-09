---
title: "From EM To VBEM"
date: 2021-07-09T18:27:01+08:00
draft: false
weight: 1

categories: ["machine learning", "statistics"]
series: []
tags: ["EM", "machine learning", "statistics", "bayes"]

ShowToc: true
TocOpen: true

cover:
    image: "img/just_imgs/arch.jpg"
    relative: false
---

## 1. Introduction

When we use K-Means or GMM to solve clustering problem, the most important hyperparameter is the number of the cluster. It is quite hard to decide and cause the good/bad performance significantly. In the mean time, K-Means also cannot handle unbalanced dataset well. However, the variational Bayesian Gaussian mixture model(VB-GMM) can solve these. VB-GMM is a Bayesian model that contains priors over the parameters of GMM. Thus, VB-GMM can be optimized by variational Bayesian expectation maximization(VBEM) and find the optimal cluster number automatically. Further, VB-GMM can also deal with the unbalanced dataset well. In this article, we will first derive the general form of the EM algorithm and prove that the EM algorithm approximates the MLE actually. In the section 2, we will introduce the variational lower bound(a.k.a evidence lower bound / VLBO / ELBO), combine EM and ELBO and, derive the variational Bayesian expectation maximization(VBEM).

## 2. Expectation Maximization

### 2.1 Naive EM

EM algorithm is useful for the model containing latent variables $Z$ when the maximum likelihood is hard to derive from the observed data $Y$. We can write the maximum likelihood of $Y$ like following

$$
\arg \max_{\theta} \mathcal{L}(Y; \theta) = \arg \max_{\theta} log(p(Y; \theta))
$$

The Expectation Maximization rewrites the question as the following

$$
\arg \max_{\theta} \mathcal{L}(Y; \theta) = \arg \max_{\theta} \ log \int_{Z} p(Y, Z; \theta) dZ
$$

Thus, we can derive the EM with an approximation $q(Z; \gamma)$ for $p(Z|Y)$ to avoid evaluating such complex distribution directly

$$
= \arg \max_{\theta} \ log \int_{Z} \frac{q(Z; \gamma)}{q(Z; \gamma)} p(Y, Z; \theta) dZ
$$

$$
= \arg \max_{\theta} \ log \ \mathbb{E}_{q} [\frac{p(Y, Z; \theta)}{q(Z; \gamma)}]
$$

Since the $log$ function is concave,` $log(\mathbb{E}_{p}[X]) \geq \mathbb{E}_{p}[log(X)]$` with Jensen's inequality.

$$
\geq \arg \max_{\theta} \ \mathbb{E}_{q} [log(\frac{p(Y, Z; \theta)}{q(Z; \gamma)})]
$$

$$
= \arg \max_{\theta} \ \int_Z q(Z; \gamma) log \ p(Y, Z; \theta) dZ - \int_Z q(Z; \gamma) log \ q(Z; \gamma) dZ
$$

$$
= \arg \max_{\theta} \ \int_Z q(Z; \gamma) log \ p(Y, Z; \theta) dZ - H_q[Z]
$$

Where $H_q[Z]$ is the entropy of $Z$ over distribution $q$

**Pseudo Code of Naive EM Algorithm**

So far, we can express the EM algorithm in a simpler way as

---

Iterate until $\theta$ converge
- E Step
  
  Evaluate $q(Z; \gamma) = p(Z|Y)$
- M Step
  
  $\arg \max_{\theta} \ \int_Z q(Z; \gamma) log \ p(Y, Z; \theta) dZ$

---

### 2.2 EM In General Form

Actually, we can represent the EM algorithm with variational lower bound $\mathcal{L}(\theta, \gamma)$

$$
\mathcal{L}(\theta, \gamma) = \mathbb{E}_{q} [log(\frac{p(Y, Z; \theta)}{q(Z; \gamma)})]
$$

$$
= \int_Z q(Z; \gamma)log \ \frac{p(Y, Z; \theta)}{q(Z; \gamma)} dZ
$$

$$
= - \int_Z q(Z; \gamma)log \ \frac{q(Z; \gamma)}{p(Z|Y)p(Y; \theta)} dZ
$$

$$
= log \ p(Y; \theta) - \int_Z q(Z; \gamma) \ log \ \frac{q(Z; \gamma)}{p(Z|Y)} dZ
$$

$$
= log \ p(Y; \theta) - KL[q(Z; \gamma) || p(Z|Y)]
$$

$$
= \mathcal{L}(Y; \theta) - KL[q(Z; \gamma) || p(Z|Y)]
$$

Thus

$$
\arg \max_{\theta} \mathcal{L}(Y; \theta) \geq \arg \max_{\theta, \gamma} \mathcal{L}(\theta, \gamma)
$$

With KKT, the constrained optimization problem can be solve with Lagrange multiplier

$$
\arg \max_{\theta, \gamma} \mathcal{L}(\theta, \gamma) = \arg \max_{\theta, \gamma} log \ p(Y; \theta) - \beta KL[q(Z; \gamma) || p(Z|Y)]
$$

Since we've known the KL-divergence is always greater or equal to 0, when $KL[q(Z; \gamma) || p(Z|Y)] = 0$, the result of EM algorithm will be equal to the maximum likelihood $\mathcal{L}(\theta, \gamma) = \mathcal{L}(Y; \theta)$. In the mean time, minimizing the KL-divergence is actually find the best approximation $q(Z; \gamma)$ for $p(Z|Y)$. 

**Pseudo Code of General EM Algorithm**

Thus, we can also represent the EM algorithm as

---

Iterate until $\theta$ converge
- E Step at k-th iteration
  
  $\gamma_{k+1} = \arg \max_{\gamma} \mathcal{L}(\theta_{k}, \gamma_{k})$
- M Step at k-th iteration
  
  $\theta_{k+1} = \arg \max_{\theta} \mathcal{L}(\theta_{k}, \gamma_{k+1})$

---

### 2.3 Variational Bayesian Expectation Maximization(VBEM)

In EM, we approximate a posterior $p(Y, Z; \theta)$ without any prior over the parameters $\theta$. Variational Bayesian Expectation Maximization(VBEM) defines a prior $p(\theta; \lambda)$ over the parameters. Thus, VBEM approximates the bayesian model $p(Y, Z, \theta; \lambda) = p(Y, Z|\theta) p(\theta; \lambda)$. Then, we can define a lower bound on the log marginal likelihood 

$$
log \ p(Y) = log \int_{Z, \theta} p(Y, Z, \theta; \lambda) dZ d\theta
$$

$$
= log \int_{Z, \theta} q(Z, \theta; \phi^{Z}, \phi^{\theta}) \frac{p(Y, Z |\theta) p(\theta; \lambda)}{q(Z, \theta; \phi^{Z}, \phi^{\theta})} dZ d\theta
$$

With mean field theory, we factorize $q$ into a joint distribution $q(Z, \theta; \phi^{Z}, \phi^{\theta}) = q(Z; \phi^{Z}) q(\theta; \phi^{\theta})$. Thus, the equation can be rewritten as

$$
= log \int_{Z, \theta} q(Z; \phi^{Z}) q(\theta; \phi^{\theta}) \frac{p(Y, Z |\theta) p(\theta; \lambda)}{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} dZ d\theta
$$

$$
= log \ \mathbb{E}_{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} [\frac{p(Y, Z |\theta) p(\theta; \lambda)}{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})}]
$$

Since the $log$ function is concave, `$log(\mathbb{E}_{p}[X]) \geq \mathbb{E}_{p}[log(X)]$` with Jensen's inequality

$$
\geq \mathbb{E}_{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} [log \  \frac{p(Y, Z |\theta) p(\theta; \lambda)}{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})}]
$$

Thus, we get the ELBO $\mathcal{L}(\phi^{Z}, \phi^{\theta})$

$$
\mathcal{L}(\phi^{Z}, \phi^{\theta}) = \mathbb{E}_{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} [log \  \frac{p(Y, Z |\theta) p(\theta; \lambda)}{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})}]
$$

Recall that we need to solve $\arg \max_{\phi^{Z}} \mathcal{L}(\phi^{Z}, \phi^{\theta})$ and $\arg \max_{\phi^{\theta}} \mathcal{L}(\phi^{Z}, \phi^{\theta})$ separately in E-step and M-step. Thus, we can derive

$$
\nabla_{\phi^Z} \mathcal{L}(\phi^{Z}, \phi^{\theta}) = 0
$$

$$
\nabla_{\phi^{\theta}} \mathcal{L}(\phi^{Z}, \phi^{\theta}) = 0
$$

Then, we can derive further

$$
\nabla_{\phi^Z} \mathcal{L}(\phi^{Z}, \phi^{\theta}) = \nabla_{\phi^Z} \mathbb{E}_{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} \Big[ log \ \frac{p(Y, Z |\theta) p(\theta; \lambda)}{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} \Big]
$$

$$
= \nabla_{\phi^Z} \mathbb{E}_{q(Z; \phi^{Z}) q(\theta; \phi^{\theta})} \Big[ log \ p(Y, Z |\theta) + log \ p(\theta; \lambda) - log \ q(Z; \phi^{Z}) - log \ q(\theta; \phi^{\theta}) \Big]
$$

$$
= \nabla_{\phi^Z} \mathbb{E}_{q(Z; \phi^{Z})} \Big[ \mathbb{E}_{q(\theta; \phi^{\theta})} [ log \ p(Y, Z | \theta) ] - log \ q(Z; \phi^{Z}) \Big] = 0
$$

Then, we can solve the equation for 0 derivative with respect to $\phi^Z$ yields the condition.

$$
\nabla_{\phi^Z} \mathbb{E}_{q(Z; \phi^{Z})} [ log \ q(Z; \phi^{Z}) ] = \nabla_{\phi^Z} \mathbb{E}_{q(Z; \phi^{Z})} \Big[ \mathbb{E}_{q(\theta; \phi^{\theta})} [ log \ p(Y, Z | \theta) ] \Big]
$$

$$
q(Z; \phi^Z) \propto e^{\mathbb{E}_{q(Z; \phi^{Z})} [ log \ q(Z; \phi^{Z}) ]}
$$

The solution for $\phi^{\theta}$ is similar

$$
\nabla_{\phi^{\theta}} \mathcal{L}(\phi^{Z}, \phi^{\theta}) = \nabla_{\phi^Z} \mathbb{E}_{q(\theta; \phi^{\theta})} \Big[ \mathbb{E}_{q(Z; \phi^{Z})} [ log \ p(Y, Z | \theta) ] + log \ p(\theta; \lambda) - log \ q(\theta; \phi^{\theta}) \Big] = 0
$$

$$
q(\theta; \phi^{\theta}) \propto e^{(\mathbb{E}_{q(Z; \phi^{Z})} [log \ p(Y, Z, \theta)])}
$$

**Pseudo Code of Variational Bayesian EM Algorithm**

---

Iterate until $\mathcal{L}(\phi^Z, \phi^{\theta})$ converge
- E Step: Update the variational distribution on $Z$
  
  $q(Z; \phi^{Z}) \propto e^{(\mathbb{E}_{q(\theta; \phi^{\theta})} [log \ p(Y, Z, \theta)])}$
- M Step: Update the variational distribution on $\theta$
  
  $q(\theta; \phi^{\theta}) \propto e^{(\mathbb{E}_{q(Z; \phi^{Z})} [log \ p(Y, Z, \theta)])}$

---