---
title: "Jax, Flax, and Optax Installation Guide"
date: 2023-01-04T17:21:30+08:00
draft: false
author: SY Chou
weight: 1

description: "Useful tricks for PyTorch"
categories: ["machine learning"]
series: []
tags: ["deep learning", "PyTorch"]

keywords:
- deep learning
- PyTorch

cover:
    image: "img/just_imgs/iced_land.jpg"
    relative: false # To use relative path for cover image, used in hugo Page-bundles
---

```bash
pip install -q --upgrade pip
pip install -q jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -q git+https://github.com/google/flax
pip install -q git+https://www.github.com/google/neural-tangents
```