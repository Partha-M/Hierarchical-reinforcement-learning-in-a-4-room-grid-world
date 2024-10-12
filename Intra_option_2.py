import numpy as np
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import RoomGridworld                  # Import gridworld environment

# ****************************** Loading Environment ****************************************
env = gym.make('RoomGridWorld-v0')

# *********************************  Parameter Initialization*********************************
alpha = 0.5
alpha_opt = 0.5
gamma = 0.9
episode = 1000
epsilon = 0.5
min_epsilon = 0.0
epsilon_opt = 0.1
max_step = 500
loop_times = 1

avg_reward_in_episodes = np.zeros(episode)
avg_steps_in_episodes = np.zeros(episode)

reduction = (epsilon - min_epsilon) / episode  # decrease exploration with episode

total_avg_steps = 0
total_avg_reward = 0


def main():
    Q_avg = np.zeros([env.observation_space.n, env.action_space.n + 2])
    Q_value = np.zeros([11, 11, env.action_space.n + 2])
    for k in range(loop_times):
        Q = np.zeros([env.observation_space.n, env.action_space.n + 2])  # Q table initialization
        Q_opt = np.zeros([2, env.observation_space.n, env.action_space.n])
        total_reward = 0
        total_steps = 0
        epsilon = 0.5
        for i in range(episode):
            # Initialize parameters
            done = False
            reward = 0
            step = 0
            state = env.reset()  # resetting episode
            while not done:
                if np.random.random() < 1 - epsilon:  # with 1-epsilon probability choose maximizing action
                    option = np.argmax(Q[state, :])
                else:
                    option = np.random.randint(0, 6)  # random action for exploration

                if option in {0, 1, 2, 3}:
                    next_state, reward, done, _ = env.step(option)
                    step += 1
                    delta = (reward + gamma * np.max(Q[next_state, :]) - Q[state, option])  # updating Q table
                    Q[state, option] = Q[state, option] + alpha * delta
                else:
                    if option == 4:
                        option_index = 0
                        if 0 <= state <= 25:
                            terminal_state = {103, 36}
                            target_state = 103
                        elif 26 <= state <= 56:
                            terminal_state = {25, 59}
                            target_state = 25
                        elif 57 <= state <= 77:
                            terminal_state = {56, 97}
                            target_state = 56
                        else:
                            terminal_state = {77, 21}
                            target_state = 77
                    else:
                        option_index = 1
                        if 25 <= state <= 55:
                            terminal_state = {56, 14}
                            target_state = 56
                        elif 56 <= state <= 76:
                            terminal_state = {77, 53}
                            target_state = 77
                        elif 77 <= state <= 102:
                            terminal_state = {103, 67}
                            target_state = 103
                        else:
                            terminal_state = {25, 79}
                            target_state = 25
                    while True:
                        if np.random.random() < 1 - epsilon:  # with 1-epsilon probability choose maximizing action
                            option_action = np.argmax(Q_opt[option_index, state, :])
                        else:
                            option_action = np.random.randint(0, 4)  # random action for exploration
                        next_state, reward, done, _ = env.step(option_action)
                        step += 1
                        reward1 = -1
                        if next_state in terminal_state:
                            Q_hat = np.max(Q[next_state, :])
                            # if next_state == target_state:
                            #     reward1 = 1
                        else:
                            Q_hat = Q[next_state, option]
                        delta_opt = (reward1 + gamma * np.max(Q_opt[option_index, next_state, :])
                                     - Q_opt[option_index, state, option_action])
                        Q_opt[option_index, state, option_action] = Q_opt[option_index, state, option_action] \
                                                                    + alpha_opt * delta_opt

                        delta = (reward + gamma * Q_hat - Q[state, option])  # updating Q table
                        Q[state, option] = Q[state, option] + alpha * delta
                        Q[state, option_action] += alpha * (reward + gamma * np.max(Q[next_state, :])
                                                            - Q[state, option_action])

                        if done or (next_state in terminal_state):
                            break
                        # print(state)
                        state = next_state

                print(state, option)
                state = next_state
                if step >= max_step:  # stop if maximum step size is reached
                    done = True
                if done:
                    print('*************************************')
                    break
            avg_reward_in_episodes[i] += reward
            avg_steps_in_episodes[i] += step

            if epsilon > min_epsilon:  # decreasing epsilon for less exploration
                epsilon -= reduction
        Q_avg += Q
        print(k)
    Q_avg = np.divide(Q_avg, loop_times)

    for index in range(104):
        if 0 <= index < 25:
            index1 = index - 0
            row = index1 // 5
            column = index1 % 5
            xcoord = row
            ycoord = column
        elif index == 25:
            xcoord = 2
            ycoord = 5
        elif 26 <= index < 56:
            index1 = index - 26
            row = index1 // 5
            column = index1 % 5
            xcoord = row
            ycoord = column + 6
        elif index == 56:
            xcoord = 6
            ycoord = 8
        elif 57 <= index < 77:
            index1 = index - 57
            row = index1 // 5
            column = index1 % 5
            xcoord = row + 7
            ycoord = column + 6
        elif index == 77:
            xcoord = 9
            ycoord = 5
        elif 78 <= index < 103:
            index1 = index - 78
            row = index1 // 5
            column = index1 % 5
            xcoord = row + 6
            ycoord = column
        else:
            xcoord = 5
            ycoord = 1

        coord = [xcoord, ycoord]
        Q_value[xcoord, ycoord, :] = Q_avg[index, :]

    plt.figure(figsize=(12, 8))
    total_options = env.action_space.n + 2
    for a_idx in range(6):
        plt.subplot(2, 3, a_idx+1)
        sns.heatmap(Q_value[:, :, a_idx], cmap="YlOrBr",
                        vmin=np.min(Q_value), vmax=np.max(Q_value))
        # Get direction name from dictionary
        # direction = [i for i in env.action_dict if env.action_dict[i] == a_idx]
        # plt.title('Q-Values for Moving {}'.format(direction[0]))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

