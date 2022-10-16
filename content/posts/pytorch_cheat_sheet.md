---
title: "Pytorch Cheat Sheet"
date: 2022-10-16T19:43:11+08:00
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

## Set Up Visible Devices For PyTorch

It's well-known that we can use ```os.environ["CUDA_VISIBLE_DEVICES"]="1,2"``` to determine which GPU can be used by the program, but as for PyTorch, most of answers says that there is no way to set visible devices in the python code for PyTorch.

However, I found ``os.environ.setdefault`` can do this.

```python
import os
import torch

gpus = [1, 2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
```

The above code will show

```bash
PyTorch detected number of availabel devices: 2
```

