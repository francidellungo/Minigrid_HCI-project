# Inverse Reinforcement Learning on Minigrid

The aim of this project is to provide a tool to train an agent on Minigrid. The human player can make game demonstrations and then the agent is trained from these demonstrations using Inverse Reinforcement Learning techniques.

The IRL algorithms are based on the following paper:
*Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations*
[[1]](#Trex).


## Code structure

## MiniGrid environment


Gym-minigrid [[2]](#minigrid) is a minimalistic gridworld package for OpenAI Gym.

There are many different environments, you can see some examples below. 

The red triangle represents the agent that can move within the environment, while
the green square (if any) represents the goal. 
There may also be other objects that the agent can interact with (doors, keys, etc.) each with a different color.

![Alt Text](./figures/minigrid.png "Minigrid environments")


## Graphical Application

### Initial window
![Alt Text](./figures/envsList.png "Available environments")

### Agents management
![Alt Text](./figures/agents.png "All the created agents")

### New agent
![Alt Text](./figures/newAgent.png "Create a new agent")

### Agent details
![Alt Text](./figures/agentDetail.png "Agent details")


## Experiments & results

## Run the project
- go to the directory in which you have downloaded the project
- go inside Minigrid_HCI-project folder with the command: `cd Minigrid_HCI-project`
- run the application with the command `python agents_window.py`


## References
<a id="Trex">[1]</a>
Daniel S. Brown, Wonjoon Goo, Prabhat Nagarajan, Scott Niekum.
Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations. (Jul 2019)
[*T-REX*](https://arxiv.org/pdf/1904.06387.pdf)

<a id="minigrid">[2]</a>
Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman.
 Minimalistic Gridworld Environment for OpenAI Gym, (2018) 
 GitHub repository [Gym-minigrid](https://github.com/maximecb/gym-minigrid)