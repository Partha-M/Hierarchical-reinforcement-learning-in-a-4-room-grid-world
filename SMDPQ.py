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
gamma = 0.9
episode = 500
epsilon = 0.5
min_epsilon = 0.0
max_step = 500
loop_times = 1

avg_reward_in_episodes = np.zeros(episode)
avg_steps_in_episodes = np.zeros(episode)

reduction = (epsilon - min_epsilon) / episode  # decrease exploration with episode

total_avg_steps = 0
total_avg_reward = 0


s1_4 = {20}
s3_4 = {22, 23, 24, 25}
s1_5 = {25, 51, 52}
s3_5 = {54, 55}
s2_6 = {26, 31}
s0_6 = {41, 46, 51, 56}
s2_7 = {56, 57, 62}
s0_7 = {72}
s1_8 = {57, 58, 77}
s3_8 = {60, 61}
s1_9 = {78}
s3_9 = {77, 80, 81, 82}
s2_10 = {82, 87, 92, 103}
s0_10 = {102}
s2_11 = {4, 9}
s0_11 = {19, 24, 103}


def option_selection(state, action):
    if action in {0, 1, 2, 3}:
        next_state, reward, tau, done = option_primitive0_3(state, action)
    elif action == 4:
        if 0 <= state <= 25:
            next_state, reward, tau, done = option4(state)
        elif 26 <= state <= 56:
            next_state, reward, tau, done = option6(state)
        elif 57 <= state <= 77:
            next_state, reward, tau, done = option8(state)
        else:
            next_state, reward, tau, done = option10(state)
    else:
        if 25 <= state <= 55:
            next_state, reward, tau, done = option5(state)
        elif 56 <= state <= 76:
            next_state, reward, tau, done = option7(state)
        elif 77 <= state <= 102:
            next_state, reward, tau, done = option9(state)
        else:
            next_state, reward, tau, done = option11(state)
    return next_state, reward, tau, done


def option_primitive0_3(state, action):
    tau = 0
    next_state, reward, done, _ = env.step(action)  # get reward and next action
    tau += 1
    return next_state, reward, tau, done


def option4(state):
    tau = 0
    reward = 0
    while state != 103 and state != 36:
        if state in s1_4:
            action = 1
        elif state in s3_4:
            action = 3
        else:
            action = 2
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option5(state):
    tau = 0
    reward = 0
    while state != 56 and state != 14:
        if state in s1_5:
            action = 1
        elif state in s3_5:
            action = 3
        else:
            action = 2
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option6(state):
    tau = 0
    reward = 0
    while state != 25 and state != 59:
        if state in s2_6:
            action = 2
        elif state in s0_6:
            action = 0
        else:
            action = 3
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option7(state):
    tau = 0
    reward = 0
    while state != 77 and state != 53:
        if state in s2_7:
            action = 2
        elif state in s0_7:
            action = 0
        else:
            action = 3
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option8(state):
    tau = 0
    reward = 0
    while state != 56 and state != 97:
        if state in s1_8:
            action = 1
        elif state in s3_8:
            action = 3
        else:
            action = 0
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option9(state):
    tau = 0
    reward = 0
    while state != 103 and state != 67:
        if state in s1_9:
            action = 1
        elif state in s3_9:
            action = 3
        else:
            action = 0
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option10(state):
    tau = 0
    reward = 0
    while state != 77 and state != 21:
        if state in s2_10:
            action = 2
        elif state in s0_10:
            action = 0
        else:
            action = 1
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def option11(state):
    tau = 0
    reward = 0
    while state != 25 and state != 79:
        if state in s2_11:
            action = 2
        elif state in s0_11:
            action = 0
        else:
            action = 1
        next_state, reward1, done, _ = env.step(action)  # get reward and next action

        reward += (gamma ** tau)*reward1
        tau += 1
        state = next_state
        if done:
            break
    return next_state, reward, tau, done


def main():
    for k in range(loop_times):
        Q = np.zeros([env.observation_space.n, env.action_space.n + 2])  # Q table initialization
        Q_value = np.zeros([11, 11, env.action_space.n + 2])
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

                next_state, reward, tau, done = option_selection(state, option)

                delta = (reward + (gamma ** tau) * np.max(Q[next_state, :]) - Q[state, option])  # updating Q table
                Q[state, option] = Q[state, option] + alpha * delta
                print(state)
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
        Q_value[xcoord, ycoord, :] = Q[index, :]

    plt.figure(figsize=(12, 8))
    total_options = env.action_space.n + 2
    for a_idx in range(6):
        plt.subplot(2, 3, a_idx+1)
        sns.heatmap(Q_value[:, :, a_idx], cmap="BuPu",
                        vmin=np.min(Q_value), vmax=np.max(Q_value))
        # Get direction name from dictionary
        # direction = [i for i in env.action_dict if env.action_dict[i] == a_idx]
        # plt.title('Q-Values for Moving {}'.format(direction[0]))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# avg_reward_in_episodes[:] = [x / loop_times for x in avg_reward_in_episodes]  # average reward and step calculation
# avg_steps_in_episodes[:] = [x / loop_times for x in avg_steps_in_episodes]
# x = np.linspace(1, episode + 1, episode)
# # Plot rewards and steps per episode
# # NAME MATLAB DATA FILE SAME AS LOADING FILE FOR PLOTTING IN MATLAB
# sio.savemat('Q_learning_0_9_C.mat',{'time_steps': x, 'average_rewards':avg_reward_in_episodes,'average_steps':avg_steps_in_episodes})
# plt.subplot(2, 1, 1)
# plt.plot(x, avg_reward_in_episodes)
# plt.xlabel('Episodes')
# plt.ylabel('Average rewards per episodes')
# plt.ylim(-4, 10)
# plt.subplot(2, 1, 2)
# plt.plot(x, avg_steps_in_episodes)
# plt.xlabel('Episodes')
# plt.ylabel('Average steps per episodes')
# plt.show()