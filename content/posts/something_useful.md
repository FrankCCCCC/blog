---
title: "Something Useful"
date: 2021-03-02T20:18:39+08:00
draft: true

expiryDate: 2021-03-01T20:18:39+00:00
---

A detailed guide:

[謦伊的閱讀筆記 - Win10 安裝 CUDA、cuDNN 教學](https://medium.com/ching-i/win10-%E5%AE%89%E8%A3%9D-cuda-cudnn-%E6%95%99%E5%AD%B8-c617b3b76deb)

# Install CUDA

Environment
- Windows10
- Geforce MX150

## Download Driver
[Select GPU Version](https://www.nvidia.com/download/index.aspx?lang=en-us)

[Download Geforce MX150 Driver](https://us.download.nvidia.com/Windows/461.72/461.72-notebook-win10-64bit-international-dch-whql.exe)

Once install the driver, the installer will install all CUDA Toolkit. **However, you should install the correct version that you need. Once you install the version you need the older version would be removed.**

You can check whether the driver is installed by ```nvidia-smi```.

Pytorch 1.7.1 is compatable with CUDA 9.2, 10.1, 10.2, 11.0.

## The way to download CUDA Toolkit

You can check whether the CUDA is installed by ```nvcc  --version```.
  
[Select OS and Version](https://developer.nvidia.com/cuda-downloads)

[Download CUDA Toolkit 10.2 for Windows 10](https://developer.nvidia.com/cuda-10.2-download-archive)

[Download CUDA Toolkit 11.2 for Windows 10](https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_461.09_win10.exe)

## The way to download cuDNN Library

The GPU driver wouldn't install cuDNN library and you need to install it manually.

You can check the CUDA version by checking the file ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\cudnn.h```.

For example, if the file was

```
define CUDNN_MAJOR 5

define CUDNN_MINOR 1

define CUDNN_PATCHLEVEL 10
```

the cuDNN version is 5.1.10
  
[Select Verions](https://developer.nvidia.com/rdp/cudnn-download)

[Download cuDNN Library 8.11 for Windows 10](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-windows-x64-v8.1.1.33.zip)

## Install TF2 GPU
[Official Site](https://www.tensorflow.org/install/gpu)

Follow the commands

```
conda create --name tf2-gpu python=3.7.9
conda activate tf2-gpu
pip install --upgrade pip
pip install tensorflow
```

Note that in my experience, Python 3.7.9 hase less compatible issues.

## Install Pytorch GPU
[](https://pytorch.org/)

```
conda create --name torch-gpu python=3.7.9
conda activate torch-gpu
pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

# SSH Guide

## On Windows

Requirement:

Install OpenSSH

Change the directory to the main directory(EX: ```C:\Users\{username}```)

Step1:

> ```ssh-keygen -t rsa -f ".\.ssh\bujo0"```

Step2: Create SSH folder under 

> ```ssh datalab@140.114.88.21 mkdir -p .ssh```
> ```cat ".\.ssh\bujo0.pub" | ssh datalab@140.114.88.21 'cat >> .ssh/authorized_keys'```

Step3:

> ```ssh -i ".\.ssh\bujo0" datalab@140.114.88.21```

## Linux

Use following script

On remote jump server

```
ssh-keygen -t rsa -f ~/.ssh/bujo-plus
ssh-copy-id -i ~/.ssh/bujo-plus ccchen@netdb-bujoplus0
ssh -i "~/.ssh/bujo-plus" ccchen@netdb-bujoplus0
```

Then, on local host(windows). Copy the private key to the local host

```
scp datalab@140.114.88.21:/home/datalab/.ssh/bujo-plus .\.ssh\
```

Connect to remote target server:

With local hostname(windows)

```
ssh -o ProxyCommand="C:\Windows\System32\OpenSSH\ssh.exe -q -W %h:%p 140.114.88.21"  ccchen@192.168.1.151 -i ".ssh/bujo-plus"
```

With IP address(windows)

```
 ssh -o ProxyCommand="C:\Windows\System32\OpenSSH\ssh.exe -q -W %h:%p 140.114.88.21"  ccchen@netdb-bujoplus0 -i ".ssh/bujo-plus"
 ```

## Something Important

Since Window seperate the directory with ```\``` but Linux ```/```, all the paths on Window should add ```" "``` across the path.

## Reference

1. [scp-ssh copy](https://blog.gtwang.org/linux/linux-scp-command-tutorial-examples/)
2. [Login SSH without password on Linux with ssh-copy-id](https://www.ibm.com/support/pages/configuring-ssh-login-without-password)
3. [ssh-copy-id on Windows](https://serverfault.com/questions/224810/is-there-an-equivalent-to-ssh-copy-id-for-windows)
4. [Alternative way to ssh-copy-id on Windows](http://www.linuxproblem.org/art_9.html)
5. [Specify ssh-keygen target file](https://superuser.com/questions/1004254/how-can-i-change-the-directory-that-ssh-keygen-outputs-to/1004263)
6. [SSH & Github](https://pjchender.github.io/2018/05/31/is-%E9%97%9C%E6%96%BC-ssh/)