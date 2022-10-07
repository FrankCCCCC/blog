---
title: "An SSH Guide"
date: 2022-10-07T17:10:32+08:00
draft: true
author: SY Chou
weight: 1

description: "Some useful SSH tricks"
categories: ["linux", "machine learning"]
series: []
tags: ["Linux", "SSH", "Github"]

keywords:
- Linux
- SSH
- Github

cover:
    image: "img/just_imgs/sand_shell.jpg"
    relative: false # To use relative path for cover image, used in hugo Page-bundles
---



## Set up SSH Login With Private Key

### On Windows

#### Requirement:

- Install OpenSSH. Please follow the instruction of this [page](https://learn.microsoft.com/zh-tw/windows-server/administration/openssh/openssh_install_firstuse)

- Change the directory to the main directory(EX: ``C:\Users\{USERNAME}``)

#### Step1: Generate SSH Key Pair

```bash {linenos=true}
ssh-keygen -t rsa -f ".\.ssh\{FILE NAME OF KEY}"
```

#### Step2: Create SSH Folder

```bash {linenos=true}
ssh {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP} mkdir -p .ssh
```

#### Step3: Copy SSH Public Key To The Remote Host

```bash {linenos=true}
cat ".\.ssh\{FILE NAME OF KEY}.pub" | ssh {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP} 'cat >> .ssh/authorized_keys'
```

#### Step4: Login To The Remote Host With SSH Key

```bash {linenos=true}
ssh -i ".\.ssh\{FILE NAME OF KEY}" {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP}
```

#### VSCode Config

After finishing the above instruction, type the login SSH command ``ssh -i ".\.ssh\{FILE NAME OF KEY}" {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP}`` to the VSCode and VSCode will generate the following configuration automatically.

```yaml
Host {CUSTOMIZED HOST NAME}
  HostName {RAMOTE HOST NAME/IP}
  IdentityFile .\.ssh\{FILE NAME OF KEY}
  User {USERNAME}
```

If you specify the port with option ``-p``, there will be one more line ``Port {LOGIN PORT}`` in the configuration.

```yaml
Host {CUSTOMIZED HOST NAME}
  HostName {RAMOTE HOST NAME/IP}
  IdentityFile .\.ssh\{FILE NAME OF KEY}
  User {USERNAME}
  Port {LOGIN PORT}
```

### Linux

#### Step1: Generate SSH Key Pair

```bash {linenos=true}
ssh-keygen -t rsa -f ~/.ssh/{FILE NAME OF KEY}
```

#### Step2: Create SSH Folder

```bash {linenos=true}
ssh {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP} mkdir -p .ssh
```

#### Step3: Copy SSH Public Key To The Remote Host

```bash {linenos=true}
ssh-copy-id -i ~/.ssh/{FILE NAME OF KEY} {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP}
```

#### Step4: Login To The Remote Host With SSH Key

```bash {linenos=true}
ssh -i "~/.ssh/{FILE NAME OF KEY}" {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP}
```

### Use The Proxy(Linux As Proxy/Jump Server)

#### Step1: Create SSH Private Key Login On The Proxy/Jump Server

On remote proxy/jump server

```bash {linenos=true}
ssh-keygen -t rsa -f ~/.ssh/{FILE NAME OF KEY}
ssh {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP} mkdir -p .ssh
ssh-copy-id -i ~/.ssh/{FILE NAME OF KEY} {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP}
ssh -i "~/.ssh/{FILE NAME OF KEY}" {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP}
```

#### Step2: Copy The Private SSH Key To The Local Host

Then, on local host(windows). Copy the private key to the local host

```bash {linenos=true}
scp {PROXY SERVER USERNAME}@{PROXY SERVER IP}:/home/{PROXY SERVER USERNAME}/.ssh/{FILE NAME OF KEY} .\.ssh\
```

#### Step3: Connect To Remote Target Server

Connect to remote target server with windows local host.

```bash {linenos=true}
ssh -o ProxyCommand="C:\Windows\System32\OpenSSH\ssh.exe -q -W %h:%p {PROXY SERVER IP}"  {LOGIN USERNAME ON THE TARGET HOST}@{TARGET HOST NAME/IP} -i ".\.ssh\{FILE NAME OF KEY}"
```

#### VSCode Config

Again, if you type the login SSH command to the VSCode,  it will generate the following configuration automatically.

```yaml
Host {CUSTOMIZED HOST NAME}
  HostName {RAMOTE TARGET HOST NAME/IP}
  ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe -q -W %h:%p {PROXY SERVER IP}
  User {LOGIN USERNAME ON THE TARGET HOST}
  IdentityFile .\.ssh\{FILE NAME OF KEY}
```

### Something Important

Since Window seperate the directory with ```\``` but Linux ```/```, all the paths on Window should add ```" "``` across the path.

## Set up Github SSH Key

### Step1: Generate SSH key

Type command

```bash {linenos=true}
ssh-keygen
```

If you want to specify some features, you can use the following options

- ``-t``: Specify cryptosystem
- ``-b``: Specify the number of bits of the key
- ``-C``: Specify the comment

```bash {linenos=true}
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

Then, follow the guide to complete the setting.

```bash {linenos=true}
Generating public/private rsa key pair.
Enter a file in which to save the key (/Users/you/.ssh/id_rsa): [Press enter to save the key in 'is_rsa' or specify other files]
Enter passphrase (empty for no passphrase): [Type a passphrase]
Enter same passphrase again: [Type passphrase again]
```

### Step2: Store Keys Into SSH Agent

Launch the SSH agent in the background.

```bash {linenos=true}
eval "$(ssh-agent -s)"
```

Add keys into the SSH agent.

```bash {linenos=true}
ssh-add -k ~/.ssh/id_rsa
```

If you don't save the key in the default file ``~/.ssh/id_rsa``, please replace the path ``~/.ssh/id_rsa`` with the the custom file.

### Step3: Copy Public Key To Create Github SSH Keys

Show the public key

```bash {linenos=true}
cat ~/.ssh/id_rsa
```

The you may see your public key

```bash {linenos=true}
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCeq5RobPZFGcoX+SIAHNghDNp1YttnANhj6gPiKwa9TN47gYmQaPZoFJJXBa5eJpLjzR8hif+4CPuqD1+xeKzQCTQ63Bg911kGHHW3RNo7PFo86hSh9yaYhGE7dD/oYsixqJnbe/ytk0SkwE8qOVkxg9o/c0S0bJOvbMr0hHNt6O8OPWFFsnFHZaY27xJv1NSjn7Q+P93sNxitviQQcYRlK8t5tbWKuF7O8WTCUz6al1iJ5SvX08BRO5TqH0lqGEkY34Lr1M2iBe1Km/ev7fZWPMs3RMSy192lDRcrBcaNF8Kgji2CxQ++GSsZ8usIUjbcywjuDS1rj3XGmi3f56/l your_email@example.com
```

Copy the content and paste it to the Github SSH keys.

![](/blog/img/an_ssh_guide/ssh_github.png)

Then, click ``Add SSH Key`` to add the SSH key.

### Step4: Test SSH Keys

```bash {linenos=true}
ssh -T git@github.com
```

If you see the following message, that means you've added the SSH keys to Github successfully.

```bash {linenos=true}
Hi FrankCCCCC! You've successfully authenticated, but GitHub does not provide shell access.
```

# Reference

- [scp-ssh copy](https://blog.gtwang.org/linux/linux-scp-command-tutorial-examples/)
- [Login SSH without password on Linux with ssh-copy-id](https://www.ibm.com/support/pages/configuring-ssh-login-without-password)
- [ssh-copy-id on Windows](https://serverfault.com/questions/224810/is-there-an-equivalent-to-ssh-copy-id-for-windows)
- [Alternative way to ssh-copy-id on Windows](http://www.linuxproblem.org/art_9.html)
- [Specify ssh-keygen target file](https://superuser.com/questions/1004254/how-can-i-change-the-directory-that-ssh-keygen-outputs-to/1004263)
- [SSH & Github](https://pjchender.github.io/2018/05/31/is-%E9%97%9C%E6%96%BC-ssh/)
- [[Git] Git 使用常见问题](http://liuxiao.org/2017/10/git-git-%E4%BD%BF%E7%94%A8%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98/)
- [設定 Github SSH 金鑰 feat. Github SSH、HTTPS 的差異](https://ithelp.ithome.com.tw/articles/10205988)
<!-- - [FrankCCCCC/useful_commands](https://github.com/FrankCCCCC/useful_commands) -->