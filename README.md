![](https://raw.githubusercontent.com/Iskandor/MotivationModels/master/assets/pythia_logo.png)
## Seer for reinforcement learning agents

Repository contains reinforcement learning agents (A2C,DQN,DDPG,PPO), different forward models and artificial motivation modules from the knowledge-based category of intrinsic motivation. Models are being tested on robotic environments and Atari games.

### Installation

Prerequisites are c++ compiler and swig
then proceed with installation:
```sh
$ git clone https://github.com/Iskandor/MotivationModels.git
$ cd MotivationModels
$ pip install -r requirements.txt
```

### Usage
```sh
$ python main.py --algorithm alg_name --env env_name --config config_id
```
Avaialable algorithms are ppo, ddpg, a2c, dqn

Environment names and ids of configuration can be found in config/alg_name.config.json 

eg. [ddpg.config.json](https://github.com/Iskandor/MotivationModels/blob/master/config/ddpg.config.json)

### Development
Project is still in development and we are planning to implement additional models and methods of reinforcement learning and intrinsic motivation. 

### Author
Matej Pechac is doctoral student of informatics specializing in the area of reinforcement learning and intrinsic motivation
- univeristy webpage: http://dai.fmph.uniba.sk/w/Matej_Pechac/en
- contact: matej.pechac@gmail.com