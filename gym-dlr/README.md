# gym-dlr

The Dynamic Load Redistribution environment is a single agent environment, featuring continuous state and action spaces which can be used to monitor and tame performance hotspots in Cloud Storage. Currently, one task is supported:

## Dlr

The Dlr task initializes a single Affinity Controller agent and rewards a score between 0 and 1 based on the Response Time(COSBENCH) performance metrics of the cluster. In order to get a perfect reward, the agent will need to know how to change affinities of each node, based on the system stats(VMSTATS) of each node. Here a perfect reward will mean the best performance that a cluster could achieve at a given state. Hence, the relative nature of the reward score makes it a difficult task to achieve.

# Installation

```bash
cd gym-dlr
pip install -e .
or
sudo -H pip3 install -e .
```

# Usage

## Method 1:

required: inside gym-dlr directory

```python
import gym
import gym_dlr
env = gym.make('Dlr-v0')
```

## Method 2:

required: Add path of gym-dlr directory to PYTHONPATH

```python
import gym
import gym_dlr
env = gym.make('Dlr-v0')
```

## Notes:

Keep COSBENCH duration > 360s
