# Motivation models

Repository contains reinforcement learning agents (A2C,DQN,DDPG,PPO) and artificial motivation modules from the knowledge-based category of intrinsic motivation. Models are being tested on robotic environments and Atari games.

### Motivation approaches
##### Prediction error motivation

Model predicting next state:

![](https://raw.githubusercontent.com/Iskandor/MotivationModels/master/assets/fm.png)

Model predicting the difference between consecutive states:

![](https://raw.githubusercontent.com/Iskandor/MotivationModels/master/assets/rfm.png)

##### Predictive surprise motivation
![](https://raw.githubusercontent.com/Iskandor/MotivationModels/master/assets/mcg.png)

### Installation

```sh
$ git clone https://github.com/Iskandor/MotivationModels.git
$ cd MotivationModels
$ pip install -r requirements.txt
```

Motivation models requires additionl gym environments:
* [pybullet-gym](https://github.com/benelot/pybullet-gym)
* [bullet3](https://github.com/bulletphysics/bullet3)
* [gym-aeris](https://github.com/michalnand/gym-aeris)

All have to be cloned and installed following their instructions.

### Usage
```sh
$ python main.py --env env_name --config config_id
```
Environment names and ids of configuration can be found in [config.json](https://github.com/Iskandor/MotivationModels/blob/master/config.json)

### Development
Project is still in development and we are planning to implement additional models and methods of reinforcement learning and intrinsic motivation. 

### Author
Matej Pechac is doctoral student of informatics specializing in the area of reinforcement learning and intrinsic motivation
- univeristy webpage: http://dai.fmph.uniba.sk/w/Matej_Pechac/en
- contact: matej.pechac@gmail.com