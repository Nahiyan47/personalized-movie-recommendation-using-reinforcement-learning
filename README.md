In this project, we build a personalized movie
recommendation system using reinforcement learning, a type of
machine learning where an agent learns by interacting with an
environment and receiving feedback in the form of rewards. We
frame the recommendation task as a Markov Decision Process
(MDP), which helps us model the decision-making process over
time. In this setup, the environment is a grid where each cell
represents a movie genre, and the agent moves through this grid
to explore and recommend movies. Each user has their own agent
that uses the Q-learning algorithm to learn which genres lead to
highly rated movies. When the agent recommends a movie and
the user has rated it positively, it receives a reward. Over time,
the agent uses this reward feedback to learn better paths in the
genre grid, helping it make smarter recommendations.
We use a subset of the MovieLens 32M dataset, focusing on
top users and their ratings to train and test our system. Our
goal is to personalize recommendations based on each user’s
individual preferences. We measure the system’s performance
using Recall@10, which checks how often the recommended
movies match the user’s actual preferences. The results show that
our reinforcement learning approach is effective—agents improve
their recommendations over time by learning which genres each
user prefers. This work shows that modeling recommendation
systems as MDPs is a promising strategy for building intelligent,
personalized recommenders.
