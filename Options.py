import numpy as np
import gym
import matplotlib.pyplot as plt
import scipy.io as sio

option = 0
gamma = 0.9

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
