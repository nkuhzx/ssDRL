# ssDRL
This repository contains the codes for our [International Journal of Social Robotics paper](https://sites.google.com/view/ssdrl2020): 
Crowd-comfort Robot Navigation among Dynamic Environment Based on social-stressed deep reinforcement learning (ssDRL)

## Introduction
Robot navigation in a dynamic environment needs to consider the comfort of the surrounding pedestrians under the premise of ensuring the safety of human, which is a challenging task. 
This paper proposes the concept of social stress index based on tension space of robot and human, which is an important part of Human-Robot interaction. Especially, the proposed approach develops crowd-comfort navigation by combining social stress index with a deep reinforcement learning framework and the value network. 
A set of typical simulation experiments show that our method effectively improves the comfort of surrounding pedestrians during the process of robot navigation.


## Prerequisites
- [Python-RVO2 library](https://github.com/sybrenstuvel/Python-RVO2)
- Python>=3.5.0
- Pytorch>=1.5.0
- torchvision>=0.6.0
- opencv3>=3.1.0
- numpy>=1.14.2

## Instruction
We set up two different environments, the first is a simple mode (env.config) and the second is a difficult mode.

- Simple mode: only 5 persons in scene and the sensing range of the robot is not considered
- Difficult mode : 10/20 persons in scene and the sensing range of the robot is set to 3m

We first train in simple mode and then fine-tuned on the difficult mode

```
python train.py --policy ssdrl --train_stage 1 --gpu True
```
After get model parameters file rl_model.pth in data/output then
```
python train.py --policy ssdrl --train_stage 2 --gpu True
```
For evaluation in stage one (simple mode)
use --visualize True can visualize the case
```
python test.py  --train_stage 1 --gpu True --visulaize True
```
For evaluation in stage one (simple mode)
```
python test.py  --train_stage 2 --gpu True
```

## Ackonwledgement

In this project, some codes for environment simulation and evaluation are built upon [ICRA2019-Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://github.com/vita-epfl/CrowdNav)