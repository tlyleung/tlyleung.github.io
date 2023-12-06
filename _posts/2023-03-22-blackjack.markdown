---
layout: post
title: "Unraveling the Secrets of Blackjack: The Journey to Basic Strategy"
description: Deriving Blackjack basic strategy using tabular solution methods from reinforcement learning methods.
image: /assets/images/blackjack/splash.png
authors: [tlyleung]
permalink: blackjack
---

# Unraveling the Secrets of Blackjack: The Journey to Basic Strategy

Blackjack, often referred to as 21, is one of the most popular casino games worldwide. Its roots can be traced back to the 17th century, with a similar game called "Vingt-et-Un" played in French casinos. However, the modern game as we know it, along with the quest to crack its code of strategy, began in the United States in the early 20th century.

The objective of Blackjack is deceptively simple: beat the dealer's hand without exceeding 21. For years, it was a game of intuition and luck, until the 1950s, when the first scientific approach to the game emerged. Roger Baldwin, Wilbert Cantey, Herbert Maisel, and James McDermott, known as the "Four Horsemen of Aberdeen," used calculators and probability theory to analyze the game. Their [1956 publication](https://web.williams.edu/Mathematics/sjmiller/public_html/341Fa09/handouts/Baldwin_OptimalStrategyBlackjack.pdf) in the Journal of the American Statistical Association laid the groundwork for all future efforts in understanding Blackjack strategy, presenting a *basic strategy* to minimize the house edge.

## Simulation Meets Strategy

The advent of computers brought a revolutionary turn to the Blackjack table. In 1962, Edward O. Thorp capitalized on the computing power of the IBM 704 to refine the basic strategy calculations. His book, "Beat the Dealer," was the first to include a mathematically proven, computer-analyzed basic strategy for Blackjack. Thorp's work used computer simulations to test and perfect the basic strategy, demonstrating that the house edge could not only be minimized but, under certain circumstances, players could actually gain an advantage. This publication also introduced the concept of card counting, allowing players to adjust their bets based on the composition of cards remaining in the deck. The casino industry was rocked by these revelations, prompting changes in game rules and the start of a cat-and-mouse game between casinos and players. Various Blackjack teams employing these techniques started springing up across the world, including the MIT Blackjack Team popularised by the movie [21](https://en.wikipedia.org/wiki/21_(2008_film)).


## Continually Refining the Basic Strategy

Since Thorp's revelations, the basic strategy for Blackjack has been continuously refined. The introduction of high-speed computers and sophisticated simulation software in the 1990s (such as [CVDATA](https://www.qfit.com/blackjack-simulator.htm)) made it possible to simulate billions of hands of Blackjack. This allowed researchers and analysts to fine-tune the basic strategy for different rule sets and number of decks in play. As a result, basic strategy charts became more precise and customized to specific game conditions. Today, basic strategy is recognized as the optimal way to play each hand in Blackjack. It is no longer just about the decision to hit or stand; it incorporates a variety of player decisions including doubling down, splitting pairs, and even the timing of surrender. 

## Tabular Solution Methods

In this post, we'll use a number of reinforcement learning techniques to derive the basic strategy. For a game like Blackjack, which has a relatively small and discrete state space that can fit in computer memory, simpler tabular solution methods are suitable. Each of these methods has its own advantages and can be suitable for different scenarios:

- **Policy Iteration:** This iterative dynamic programming algorithm alternates between evaluating the current policy and improving it by making it greedier. In the context of Blackjack, this method can systematically calculate the expected returns for each possible action, refining the strategy until it cannot be further improved.

- **Monte Carlo with Exploring Starts:** This technique simulates many Blackjack games, starting from random states and actions, to estimate the value of each decision. It converges to the best strategy by learning from the outcomes of each game, without needing a pre-defined model of the game's dynamics.

- **Q-Learning:** Methods like SARSA and Q-learning are part of the Temporal Difference (TD) learning family. They update the value functions based on the estimated returns rather than waiting until the end of the episode, as in Monte Carlo methods. These methods can learn optimal policies online without the need for episode completion.

# Dynamic Programming: Policy Iteration

Dynamic programming is a method used to solve complex problems by breaking them down into simpler subproblems. It is based on the principle of optimality, which states that the optimal solution to a problem can be composed of optimal solutions to its subproblems. In the context of reinforcement learning, dynamic programming techniques are used to solve Markov Decision Processes (MDPs) where the full model of the environment is known, including state transition probabilities and rewards.

Policy iteration is a dynamic programming algorithm used to find the optimal policy in an MDP. It alternates between two steps: policy evaluation and policy improvement. During policy evaluation, the value function for a given policy is calculated until it sufficiently converges (i.e., changes between iterations are below a small threshold). Policy improvement then updates the policy by making it greedier with respect to the current value function; that is, for each state, the action that yields the highest value according to the current value function is chosen. This process repeats until the policy is stable, meaning it no longer changes from one iteration to the next, indicating that the optimal policy has been found.

First, we start off with loading some helper functions from [Gymnasium's Blackjack environment](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py) implementation, even though in this section we don't use the environment itself.


```python
def cmp(a, b):
    return float(a > b) - float(a < b)


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def has_usable_ace(hand):  # does this hand have a usable ace?
    return int(1 in hand and sum(hand) + 10 <= 21)


def sum_hand(hand):  # return current hand total
    return sum(hand) + (10 if has_usable_ace(hand) else 0)


def is_bust(hand):  # is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # what is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]
```

Then, through simulation, we precompute the distribution of possible outcomes for each of the dealer's potential starting cards. This is used when the player stands or doubles and the dealer plays out their hand.


```python
import numpy as np


probabilities = np.zeros((11, 22))
num_simulations = 1_000_000
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

for _ in range(num_simulations):
    dealer_card_value = int(np.random.choice(deck))
    dealer_cards = [dealer_card_value, int(np.random.choice(deck))]
    while sum_hand(dealer_cards) < 17:
        dealer_cards.append(int(np.random.choice(deck)))
        
    probabilities[dealer_card_value][score(dealer_cards)] += 1
    
probabilities /= probabilities.sum(axis=1, keepdims=True)
```

We're using the simplified version of Blackjack described in Example 5.1 of [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). This version of Blackjack follows the typical rules, where the objective is to obtain cards whose sum is as close to 21 as possible without going over, but we are limited to only two actions: stand and hit. The `expected_return` function lays out the complete state transition probabilities and rewards.


```python
def expected_return(state, action, V, gamma=1.0):
    player_sum, dealer_card_value, usable_ace = state
    rewards = 0.0

    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    # Get probabilities for each card in the deck
    card_probabilities = {card: deck.count(card) / len(deck) for card in set(deck)}

    if action == 0:  # stand
        dealer_scores = probabilities[dealer_card_value]
        wins = np.sum(dealer_scores[0:player_sum])
        losses = np.sum(dealer_scores[(player_sum + 1) :])
        rewards = wins - losses
    else:  # hit
        for next_card in set(deck):
            player_cards = [1, player_sum - 11] if usable_ace else [player_sum]
            player_cards.append(next_card)
            next_player_sum = sum_hand(player_cards)
            next_usable_ace = has_usable_ace(player_cards)

            if is_bust(player_cards):
                reward = -1
            else:
                next_state = (next_player_sum, dealer_card_value, next_usable_ace)
                reward = gamma * V[next_state]  # no immediate reward for hitting

            rewards += card_probabilities[next_card] * reward

    return rewards


def policy_evaluation(V, policy, theta=1e-10):
    while True:
        delta = 0.0
        for state in np.ndindex(V.shape):
            v = V[state]
            action = policy[state]
            V[state] = expected_return(state, action, V)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break


def policy_improvement(V, policy):
    policy_stable = True
    for state in np.ndindex(V.shape):
        old_action = policy[state]
        action_values = {action: expected_return(state, action, V) for action in [0, 1]}
        policy[state] = max(action_values, key=action_values.get)
        if old_action != policy[state]:
            policy_stable = False
    return policy_stable


def policy_iteration(V, policy):
    policy_stable = False
    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)
```


```python
# Initialize state-value function V and policy
V = np.zeros((32, 11, 2))
policy = np.zeros((32, 11, 2), dtype=int)

policy_iteration(V, policy)

# Build state-action-value function Q
Q1 = np.zeros((32, 11, 2, 2))

for index in np.ndindex(Q1.shape):
    *state, action = index
    Q1[index] = expected_return(state, action, V)
```

Plotting the policy $$\pi$$ and the state-action value function Q, we get:


```python
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

LABELS = [
    "Stand",
    "Hit",
    "Split",
    "Double if possible, else Stand",
    "Double if possible, else Hit",
]

def plot_state_action_values(state_action_values):
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)
    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)

    state_value_usable_ace[:13,:] = np.NaN
    state_value_no_usable_ace[:5,:] = np.NaN

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    im = ax.matshow(action_usable_ace, aspect="auto")
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Soft Player Sum')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(12.5, 21.5)
    ax.set_title('Optimal policy with usable Ace')

    norm = Normalize(vmin=0, vmax=1)
    patches = [mpatches.Patch(color=im.cmap(norm(i)), label=LABELS[i]) for i in range(2)]
    plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=1, borderaxespad=1)
    
    ax = fig.add_subplot(2, 2, 3, sharex=ax)
    ax.matshow(action_no_usable_ace, aspect="auto")
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Hard Player Sum')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(4.5, 21.5)
    ax.set_title('Optimal policy without usable Ace')

    x, y = np.meshgrid(np.arange(11), np.arange(22))

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot_surface(x, y, state_value_usable_ace[:22])
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Soft Player Sum')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(12.5, 21.5)
    ax.set_zlim(-1, 1)
    ax.set_title('Optimal value with usable Ace')

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(x, y, state_value_no_usable_ace[:22])
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Hard Player Sum')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(4.5, 21.5)
    ax.set_zlim(-1, 1)
    ax.set_title('Optimal value without usable Ace')

    plt.tight_layout()
    plt.show()
```


```python
plot_state_action_values(Q1)
```


    
![png](/assets/images/blackjack/blackjack_14_0.png){: .multiply }
    


# Gymnasium Environment

Dynamic programming methods depend on having a complete model of the environment, including all possible state transitions and rewards, which isn't always available. The next two tabular solution methods only require an interactive environment. The Blackjack environment from Gymnasium follows the same simplified version of the Blackjack card game in the previous section. 

Here's a short tutorial on how to use the Blackjack Gymnasium environment:

## Initialization

To start using the Blackjack environment, you need to import the Gymnasium package and then create and initialize the Blackjack environment:


```python
import gymnasium as gym

# Create the Blackjack environment
env = gym.make('Blackjack-v1')

# Start a new game
state, info = env.reset()

print(state)
```

    (10, 6, 0)


The `reset` method starts a new game and returns the initial game state, which typically includes the player's current sum, the dealer's one visible card, and whether or not the player has a usable ace (an ace that can be counted as 11 without busting).

## Game Dynamics

You interact with the game by calling the `step` method with an action:

- action = 0 corresponds to "stick" or "stand" (not asking for another card)
- action = 1 corresponds to "hit" (asking for another card to try to get closer to 21)

The step method returns:

- `next_state`: the new state after the action
- `reward`: the reward from the action
- `terminated`: a boolean indicating if the game is over
- `truncated`: a boolean indicating that the episode has ending due to an externally defined condition
- `info`: a dictionary with additional information, usually empty in the Blackjack environment


```python
# Example action: hit
next_state, reward, terminated, truncated, info = env.step(1)

print(next_state)
```

    (19, 6, 0)


## Ending the Game

The game continues with the player taking actions until they decide to stand or they bust (their sum exceeds 21). Once the player stands, the dealer plays with a fixed strategy (such as hitting until the dealer's sum is 17 or more). After the dealer's turn, the game ends and the done flag is set to True. The reward is then assigned: +1 for winning, 0 for a draw, and -1 for losing.

## Example Loop

Here's an example of playing one round of Blackjack with a simple strategy: hit until reaching 18 or more, then stand.


```python
import gymnasium as gym

env = gym.make('Blackjack-v1')

state, info = env.reset()

while True:
    player_sum, dealer_card, usable_ace = state
    action = 0 if player_sum >= 18 else 1   # strategy: hit until our sum is 18 or more
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final state: {state}")
print(f"Reward: {reward}")
```

    Final state: (19, 8, 0)
    Reward: 1.0


## Closing the Environment

When you're done with the environment, you should close it to free up resources:


```python
env.close()
```

# Monte Carlo: Monte Carlo with Exploring Starts

Monte Carlo methods in reinforcement learning are used to estimate the value functions of states or state-action pairs based on averaging the returns (total rewards) received from episodes. These episodes are sequences of states, actions, and rewards, from the start of an environment to a terminal state.

Monte Carlo with Exploring Starts is a specific approach within this family of methods. The key features are:

- **Exploring Starts:** To ensure that all state-action pairs are visited sufficiently, each episode begins with a randomly chosen state and action. This is crucial for the method to explore and learn the value of all possible decisions in environments where some states or actions are less likely to be visited under a certain policy.

- **No Model Needed:** Unlike dynamic programming, Monte Carlo methods don't require a complete model of the environment's dynamics. They can learn directly from experience obtained through interaction with the environment.

- **Episode-based Learning:** The value of states or state-action pairs is updated after each episode. This means that the method waits until the end of an episode to update the value estimates based on the total return received.

In Blackjack, for instance, Monte Carlo with Exploring Starts can be used to estimate the optimal policy by simulating many hands of the game, each starting with a random state (player's and dealer's cards) and an initial action, and then playing out the hand according to a given policy.


```python
import gymnasium as gym

from collections import deque
from tqdm import tqdm


# Function to generate a single episode using the current policy
def generate_episode(env, policy):
    episode = []
    state, _ = env.reset()  # choose S_0 randomly
    done = False
    while not done:
        # Select an action: choose A_0 randomly, policy-based subsequently
        action = env.action_space.sample() if not episode else policy[state]
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


def monte_carlo_es(env, gamma=0.95, num_episodes=1_000_000, deque_size=None):
    deque_size = num_episodes if deque_size is None else deque_size
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=deque_size)
    state_n = [space.n for space in env.observation_space]
    state_action_n = state_n + [env.action_space.n]
    
    Qs = deque(maxlen=deque_size)
    Q = np.zeros(state_action_n)
    state_action_returns = np.zeros(state_action_n)
    state_action_counts = np.zeros(state_action_n, dtype=int)
    policy = np.ones(state_n, dtype=int)
    
    for _ in tqdm(range(num_episodes)):
        episode = generate_episode(env, policy)
        G = 0.0  # initialize the discounted return
        visited = set()  # create a set to track visited state-action pairs

        # Loop over episode in reverse to calculate returns and update Q-values
        for state, action, reward in reversed(episode):
            G = gamma * G + reward  # update discounted return

            if (state, action) not in visited:
                visited.add((state, action))
                state_action_returns[state][action] += G
                state_action_counts[state][action] += 1
                policy[state] = np.argmax(
                    state_action_returns[state] / state_action_counts[state]
                )
                
        Q = state_action_returns / state_action_counts
        Qs.append(Q)

    env.close()
    
    return np.stack(Qs), env.return_queue, env.length_queue
```


```python
env = gym.make("Blackjack-v1")
Qs2, return_queue2, length_queue2 = monte_carlo_es(env)
Q2 = Qs2[-1]
```


```python
plot_state_action_values(Q2)
```


    
![png](/assets/images/blackjack/blackjack_28_0.png){: .multiply }
    


# Temporal Difference (TD) Learning: Q-Learning

Temporal Difference (TD) Learning, and specifically Q-Learning, is another method in reinforcement learning used for estimating the optimal policy. The key aspects are:

- **Learning from Incomplete Episodes**: TD methods, unlike Monte Carlo, do not need to wait until the end of an episode to update their value estimates. They can learn online after every step taken in the environment. This is particularly useful in environments where episodes are long or infinite.

- **Q-Learning**: This is a specific type of TD learning focused on learning the optimal action-value function, denoted as Q. Q-Learning is an off-policy learner, meaning it can learn the value of the optimal policy independently of the agent's actions. It updates its Q-values using the Bellman equation as follows:

  $$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

  Here, $$\alpha$$ is the learning rate, $$\gamma$$ is the discount factor, $$r_{t+1}$$ is the reward received after taking action $$a_t$$ in state $$s_t$$, and $$\max_{a} Q(s_{t+1}, a)$$ is the maximum estimated value of any action in the next state $$s_{t+1}$$.

- **Model-Free**: Like Monte Carlo, Q-Learning does not require a model of the environment and learns from the experience obtained through interaction.

Q-Learning is particularly effective in environments with discrete states and actions, where the goal is to learn an optimal policy that maximizes the expected return. It is robust in stochastic environments and can handle problems with a large state space.


```python
import gymnasium as gym

from collections import deque
from tqdm import tqdm


def q_learning(
    env,
    alpha=0.001,
    start_epsilon=1.0,
    end_epsilon=0.1,
    gamma=0.95,
    num_episodes=1_000_000,
    deque_size=None,
):
    deque_size = num_episodes if deque_size is None else deque_size
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=deque_size)
    state_n = [space.n for space in env.observation_space]
    state_action_n = state_n + [env.action_space.n]

    Qs = deque(maxlen=deque_size)
    Q = np.zeros(state_action_n)

    for i in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False

        epsilon = start_epsilon + (i / num_episodes) * (end_epsilon - start_epsilon)
        while not done:
            # Choose the next action using an epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore: random action
            else:
                action = np.argmax(Q[state])  # Exploit: best action based on Q-values

            # Take action and observe reward and next state
            next_state, reward, done, _, _ = env.step(action)

            # Q-Learning update
            future_q = 0 if done else np.max(Q[next_state])
            temporal_difference = reward + gamma * future_q - Q[state][action]
            Q[state][action] += alpha * temporal_difference

            # Update state
            state = next_state

        Qs.append(Q.copy())

    env.close()

    return np.stack(Qs), env.return_queue, env.length_queue
```


```python
env = gym.make('Blackjack-v1')
Qs3, return_queue3, length_queue3 = q_learning(env)
Q3 = Qs3[-1]
```


```python
plot_state_action_values(Q3)
```


    
![png](/assets/images/blackjack/blackjack_33_0.png){: .multiply }
    


Note: as can be seen in the two policy charts above, the policy is close to, but doesn't quite match the policies derived from the other two methods.

# Comparison

To aid the comparison between the two model-free approaches, the environments were wrapped in a `RecordEpisodeStatistics` wrapper. This provides us with data on how episode returns and episode lengths change as training progresses. We also plot the MSE between the two model-free methods and the ground truth policy derived through policy iteration.


```python
import pandas as pd

fig = plt.figure(figsize=(12, 4))

df_returns = pd.DataFrame({"Monte Carlo ES": np.array(return_queue2).flatten(),
                           "Q-Learning": np.array(return_queue3).flatten()})
df_returns.rolling(1_000, center=True).mean().plot(
    ax=fig.add_subplot(1, 3, 1),
    title="Episode Returns",
    xlabel="Episodes",
)

df_lengths = pd.DataFrame({"Monte Carlo ES": np.array(length_queue2).flatten(),
                           "Q-Learning": np.array(length_queue3).flatten()})
df_lengths.rolling(1_000, center=True).mean().plot(
    ax=fig.add_subplot(1, 3, 2),
    title="Episode Lengths",
    xlabel="Episodes",
)

mse2 = np.nanmean(((Qs2[:,:,1:] - Q1[:,1:])**2), axis=(1, 2, 3, 4))
mse3 = np.nanmean(((Qs3[:,:,1:] - Q1[:,1:])**2), axis=(1, 2, 3, 4))
df_mse = pd.DataFrame({"Monte Carlo ES": mse2, "Q-Learning": mse3})
df_mse.plot(
    ax=fig.add_subplot(1, 3, 3),
    title="Training MSE from Ground Truth",
    xlabel="Episodes",
    ylim=(0, 0.5),
)

plt.tight_layout()
plt.show()
```


    
![png](/assets/images/blackjack/blackjack_37_0.png){: .multiply }
    


Let's play out 1,000,000 games for each policy and compare the mean rewards. As an additional datapoint, we'll also include a policy based on the simple strategy of hitting until our sum is 18 or more, which we saw in the Gymnasium Environment section.


```python
# Simple strategy: hit until our sum is 18 or more
simple_policy = np.zeros((32, 11, 2), dtype=int)
simple_policy[:19,:,:] = 1
```


```python
env = gym.make("Blackjack-v1")

policies = [
    simple_policy,
    np.argmax(Q1, axis=-1),
    np.argmax(Q2, axis=-1),
    np.argmax(Q3, axis=-1),
]
rewards = [[], [], [], []]

for _ in tqdm(range(1_000_000)):
    for i, policy in enumerate(policies):
        state, info = env.reset()

        while True:
            player_sum, dealer_card, usable_ace = state
            action = policy[player_sum, dealer_card, usable_ace]
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                rewards[i].append(reward)
                break

print(np.mean(rewards[0]))
print(np.mean(rewards[1]))
print(np.mean(rewards[2]))
print(np.mean(rewards[3]))
```

    100%|███████████████████████████████| 1000000/1000000 [07:37<00:00, 2187.18it/s]


    -0.19649
    -0.045929
    -0.043312
    -0.042268


As we see here, these are all losing strategies, although the derived policies perform markedly better than the simple strategy. In fact, this falls in line with commonly accepted house edge figures of 2% with simple strategies and 0.5% when employing Basic Strategy.

# Extending the Environment

In this section, we'll attempt to extend the simplified Blackjack version to include the double and split actions that make up the full game.

The simple Blackjack environment uses an observation space that comprises of 3 numbers:
- the player's current sum (1-31, the top of the range is when a player decides to hit a 21)
- the value of the dealer's one showing card (1-10, where 1 is ace)
- whether the player holds a usable ace (0 or 1)

To support the full game, we change the observation space to include 6 numbers:
- the value of the player's first card (1-10)
- the value of the player's second card (1-10)
- *the player’s current sum (1-31, the top of the range is when a player decides to hit a 21)*
- *the value of the dealer’s one showing card (1-10, where 1 is ace)*
- *whether the player holds a usable ace (0 or 1)*
- the number of splits taken place (0-2)

Not only is this new observation space much bigger, it also contains illegal states (e.g. the player sum may not add up to first card and second hand), which can be problematic.


```python
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

def plot_state_action_values_extended(state_action_values):
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    card_probabilities = {card: deck.count(card) / len(deck) for card in set(deck)}

    weights = np.zeros((11, 11))
    for i in range(1, 11):
        for j in range(1, 11):
            weight = card_probabilities[i] * card_probabilities[j]
            weights[i, j] = weight

    weights = weights[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    weighted_state_action_values = np.nansum(state_action_values * weights, axis=(0, 1))

    action_usable_ace = np.nanargmax(weighted_state_action_values[:, :, 1, 0, :], axis=-1)
    action_no_usable_ace = np.nanargmax(weighted_state_action_values[:, :, 0, 0, :], axis=-1)
    
    pairs = []
    for i in range(11):
        player_sum = sum_hand([i, i])
        usable_ace = 1 if i == 1 else 0
        pairs.append(state_action_values[i, i, player_sum, :, usable_ace, 0])
    action_pairs = np.argmax(np.stack(pairs), axis=-1)
    
    fig = plt.figure(figsize=(3, 15))
    
    ax1 = plt.subplot(311)
    ax1.matshow(action_no_usable_ace, aspect="auto")
    ax1.set_ylabel("Hard Player Sum")
    ax1.set_xlim(0.5, 10.5)
    ax1.set_ylim(4.5, 21.5)
    ax1.invert_yaxis()

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.matshow(action_usable_ace, aspect="auto")
    ax2.set_ylabel("Soft Player Sum")
    ax2.set_ylim(12.5, 21.5)
    ax2.invert_yaxis()
    
    ax3 = plt.subplot(313, sharex=ax1)
    im = ax3.matshow(action_pairs, aspect="auto")
    ax3.set_ylabel("Player Pair")
    ax3.set_ylim(0.5, 10.5)
    ax3.invert_yaxis()
    
    norm = Normalize(vmin=0, vmax=4)
    patches = [mpatches.Patch(color=im.cmap(norm(i)), label=LABELS[i]) for i in range(5)]
    plt.legend(handles=patches, bbox_to_anchor=(1, -0.1), loc=1, borderaxespad=0.0)

    plt.tight_layout()
```

## Dynamic Programming: Policy Iteration


```python
def expected_return(state, action, V, gamma=1.0):
    first_card, second_card, player_sum, dealer_card_value, usable_ace, splits = state
    rewards = 0.0

    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    # Get probabilities for each card in the deck
    card_probabilities = {card: deck.count(card) / len(deck) for card in set(deck)}

    if action in [3, 4]:  # double
        can_double = (
            sum_hand([first_card, second_card]) == player_sum
            and first_card > 0
            and second_card > 0
            and (
                ((usable_ace == 0) and (first_card != 1 and second_card != 1))
                or ((usable_ace == 1) and (first_card == 1 or second_card == 1))
            )
        )
        if can_double:
            for next_card in set(deck):
                player_cards = [first_card, second_card, next_card]
                next_player_sum = sum_hand(player_cards)

                if is_bust(player_cards):
                    reward = -2.0
                else:
                    dealer_scores = probabilities[dealer_card_value]
                    wins = np.sum(dealer_scores[0:next_player_sum])
                    losses = np.sum(dealer_scores[(next_player_sum + 1) :])
                    reward = wins - losses

                rewards += 2.0 * card_probabilities[next_card] * reward
        else:
            if action == 3:  # convert to stand
                action = 0
            elif action == 4:  # convert to hit
                action = 1

    if action == 0:  # stand
        dealer_scores = probabilities[dealer_card_value]
        wins = np.sum(dealer_scores[0:player_sum])
        losses = np.sum(dealer_scores[(player_sum + 1) :])
        rewards = wins - losses
    elif action == 1:  # hit
        for next_card in set(deck):
            if first_card == 0:
                player_cards = [next_card]
                next_first_card = next_card
                next_second_card = 0
            elif second_card == 0:
                player_cards = [first_card, next_card]
                next_first_card = first_card
                next_second_card = next_card
            else:
                player_cards = [1, player_sum - 11] if usable_ace else [player_sum]
                player_cards.append(next_card)
                next_first_card = first_card
                next_second_card = second_card

            next_player_sum = sum_hand(player_cards)
            next_usable_ace = has_usable_ace(player_cards)

            if is_bust(player_cards):
                reward = -1.0
            else:
                next_state = (
                    next_first_card,
                    next_second_card,
                    next_player_sum,
                    dealer_card_value,
                    next_usable_ace,
                    splits,
                )
                reward = gamma * V[next_state]  # no immediate reward for hitting

            rewards += card_probabilities[next_card] * reward
    elif action == 2:  # split
        can_split = (
            sum_hand([first_card, second_card]) == player_sum
            and first_card > 0
            and second_card > 0
            and (
                ((usable_ace == 0) and (first_card != 1 and second_card != 1))
                or ((usable_ace == 1) and (first_card == 1 or second_card == 1))
            )
            and first_card == second_card
            and splits < 2
        )
        if can_split:
            for next_card in set(deck):
                player_cards = [first_card, next_card]
                next_state = (
                    first_card,
                    next_card,
                    sum_hand(player_cards),
                    dealer_card_value,
                    has_usable_ace(player_cards),
                    splits + 1,
                )
                reward = gamma * V[next_state]
                rewards += 2.0 * card_probabilities[next_card] * reward
        else:
            rewards = -2.0  # illegal action

    return rewards


def policy_evaluation(V, policy, theta=1e-10):
    while True:
        delta = 0.0
        for state in np.ndindex(V.shape):
            v = V[state]
            action = policy[state]
            V[state] = expected_return(state, action, V)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break


def policy_improvement(V, policy):
    policy_stable = True
    for state in np.ndindex(V.shape):
        old_action = policy[state]
        action_values = {
            action: expected_return(state, action, V) for action in range(5)
        }
        policy[state] = max(action_values, key=action_values.get)
        if old_action != policy[state]:
            policy_stable = False

    return policy_stable


def policy_iteration(V, policy):
    policy_stable = False
    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)

    return V, policy
```


```python
# Initialize state-value function V and policy
V = np.zeros((11, 11, 22, 11, 2, 3))
policy = np.zeros((11, 11, 22, 11, 2, 3), dtype=int)

policy_iteration(V, policy)

# Build state-action-value function Q
Q4 = np.zeros((11, 11, 22, 11, 2, 3, 5))

for index in np.ndindex(Q4.shape):
    *state, action = index
    Q4[index] = expected_return(state, action, V)
```


```python
plot_state_action_values_extended(Q4)
```


    
![png](/assets/images/blackjack/blackjack_48_0.png){: .d-block .multiply .mx-auto .w-50 }
    


The basic strategy for the full Blackjack game derived from policy iteration corresponds well with the [Wizard of Odds' Basic Stragegy Table](https://wizardofodds.com/games/blackjack/strategy/calculator/). For some unclear reason, the split action seems to be missing from the third table.

## Monte Carlo: Monte Carlo with Exploring Starts

We try to recreate the full Blackjack game as a custom Gymnasium environment so we can apply Monte Carlo with Exploring Starts.


```python
from gymnasium import spaces


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


class BlackjackEnv(gym.Env):
    """
    Added double and split actions to Blackjack environment from:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py
    """

    def __init__(self, hit_soft_17=False, double_after_split=True):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(11),
                spaces.Discrete(11),
                spaces.Discrete(32),
                spaces.Discrete(11),
                spaces.Discrete(2),
                spaces.Discrete(3),
            )
        )
        self.hit_soft_17 = hit_soft_17
        self.double_after_split = double_after_split

    def step(self, action):
        assert self.action_space.contains(action)

        if action in [3, 4]:  # double
            can_double = len(self.player) == 2 and (
                self.splits == 0 or self.double_after_split
            )
            if can_double:
                self.player.append(draw_card(self.np_random))
                terminated = True
                if is_bust(self.player):
                    reward = -2.0
                else:
                    reward = 2.0 * self.play_dealer()
            else:
                if action == 3:  # convert to stand
                    action = 0
                elif action == 4:  # convert to hit
                    action = 1

        if action == 0:  # stand
            terminated = True
            reward = self.play_dealer()
        elif action == 1:  # hit
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        elif action == 2:  # split
            can_split = (
                self.splits < 2
                and len(self.player) == 2
                and self.player[0] == self.player[1]
            )
            if can_split:
                # Assume split hands play out the same and only follow one of the splits
                # Cannot bust on 2 cards so no need to check
                self.splits += 1
                self.player[1] = draw_card(self.np_random)
                terminated = False
                reward = 0.0
            else:
                terminated = True
                reward = -1.0

        return self._get_obs(), reward *(2**self.splits), terminated, False, {}

    def play_dealer(self):
        soft_17 = has_usable_ace(self.dealer) and sum_hand(self.dealer) == 17
        while sum_hand(self.dealer) < 17 or (soft_17 and self.hit_soft_17):
            self.dealer.append(draw_card(self.np_random))

        return cmp(score(self.player), score(self.dealer))

    def _get_obs(self):
        return (
            self.player[0],
            self.player[1],
            sum_hand(self.player),
            self.dealer[0],
            has_usable_ace(self.player),
            self.splits,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.splits = 0
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs(), {}
```


```python
env = BlackjackEnv()
Qs5, return_queue5, length_queue5 = monte_carlo_es(env, num_episodes=20_000_000, deque_size=100)
Q5 = Qs5[-1]
```


```python
plot_state_action_values_extended(Q5)
```


![png](/assets/images/blackjack/blackjack_54_0.png){: .d-block .multiply .mx-auto .w-50 }
    


Due to the observation space being so large, even after running Monte Carlo with Exploring Starts for 20,000,000 episodes, we are still far from convergence.

# References

- [Baldwin, R. R., Cantey, W. E., Maisel, H., & McDermott, J. P. (1956). The Optimum Strategy in Blackjack. Journal of the American Statistical Association.](https://web.williams.edu/Mathematics/sjmiller/public_html/341Fa09/handouts/Baldwin_OptimalStrategyBlackjack.pdf)
- [Shackleford, M. (2010). Blackjack Basic Strategy Calculator. Wizard of Odds.](https://wizardofodds.com/games/blackjack/strategy/calculator/)
- [Silver, D. (2015). Reinforcement Learning Assignment: Easy21. Lectures on 
Reinforcement Learning.](https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf)
- [Sutton, R. S. & Barto, A. G. (2020) Reinforcement Learning: An Introduction. Second edition. MIT Press.](http://incompleteideas.net/book/the-book-2nd.html)
- Thorp, E. (1966). Beat the Dealer: A Winning Strategy for the Game of Twenty-One.
- [Zemann, T. (2022). Solving Blackjack with Q-Learning. Gymnasium Documentation.](https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/)
