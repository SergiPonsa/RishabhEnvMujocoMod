import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



import os
import gym
from gym import spaces, envs
import argparse
import numpy as np
import itertools
import time
from builtins import input
import random

from mujoco_py.modder import TextureModder, MaterialModder
import cv2

def MasterProgram(size,comm):
    results = [0]*(size-1)
    done = [False]*(size-1)
    start = True
    for i in range(100):
        print("Episode ",i)
        print("send parameters to start to all")
        [comm.send(start,dest=d,tag=1) for d in range(1,size)]

        for s in range(1,size):
            print("reading data of ",s)
            results[s-1] = comm.recv(source=s,tag=2)
            done[s-1] = comm.recv(source=s,tag=3)
        print("results",results)
        print("done",done)

def MasterProgramCrossEntropy(env,size,comm,parameters_to_change,n_iterations = 700,change_sigma_every = 50,
    pop_size = 50, elite_fraction = 4/50, sigma = 0.3, sigma_reduction_per_one = 0.65,\
    per_one_parameters_values = True):

    #Numbers of elements that you keep as the better ones
    n_elite=int(pop_size*elite_fraction)
    #scores doble end queee , from iterations size * 0.1
    scores_deque = deque(maxlen=int(n_iterations*0.1))
    #intial scores empty
    scores = []
    #Save actions to see how they evolve
    best_actions = []
    #Select a seed to make the results the same every test, not depending on the seed
    np.random.seed(0)
    #Initial best weights, are from 0 to 1, it's good to be small the weights, but they should be different from 0.
    # small to avoid overfiting , different from 0 to update them

    if (per_one == True):
        original_parameters = np.array( env.Get_Mujoco_Parameters(parameters_to_change,unique_List = True) )
        best_weight = sigma*np.random.randn( len(list(original_action)) )

    else:
        original_parameters = np.array( env.Get_Mujoco_Parameters(parameters_to_change,unique_List = True) )
        best_weight = np.add(sigma*np.random.randn( len(list(original_action)) ),original_parameters)

    #Each iteration, modify  + (from 0 to 1) the best weight randomly
    #Computes the reward with these weights
    #Sort the reward to get the best ones
    # Save the best weights
    # the Best weight it's the mean of the best one
    #compute the main reward of the main best rewards ones
    #this it's show to evalute how good its

    for i_iteration in range(1, n_iterations+1):

        #Generate new population weights, as a mutation of the best weight to test them
        weights_pop = [best_weight + (sigma*np.random.randn(env.action_space)) for i in range(pop_size)]

        #Compute the parameters and obtain the rewards for each of them
        #print("iteration "+str(i_iteration))
        if (per_one == True):
            rewards=[]
            for weights in weights_pop:
                #print("New weights")
                #print(weights)
                #t.sleep(1000)
                parameters_to_change_values = list( np.add(np.multiply(weights,original_action),original_action) )
                results_parameters = [0]*(size-1)
                done = [False]*(size-1)
                #"send parameters to start to all",
                [comm.send(parameters_to_change_values,dest=d,tag=1) for d in range(1,size)]
                #wait all of them end
                for s in range(1,size):
                    # print("reading data of ",s)
                    results[s-1] = comm.recv(source=s,tag=2)
                    done[s-1] = comm.recv(source=s,tag=3)
                result_avg = numpy.average( np.array(results) )
                rewards.append(result_avg)
        else:
            rewards=[]
            for weights in weights_pop:
                parameters_to_change_values = weights
                results_parameters = [0]*(size-1)
                done = [False]*(size-1)
                #"send parameters to start to all",
                [comm.send(parameters_to_change_values,dest=d,tag=1) for d in range(1,size)]
                #wait all of them end
                for s in range(1,size):
                    # print("reading data of ",s)
                    results[s-1] = comm.recv(source=s,tag=2)
                    done[s-1] = comm.recv(source=s,tag=3)
                result_avg = numpy.average( np.array(results) )
                rewards.append(result_avg)

        print("rewards" + str(i_iteration))
        print(rewards)
        #print("\n")

        #Sort the rewards to obtain the best ones
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]

        #Set the best weight as the mean of the best ones

        best_weight = np.array(elite_weights).mean(axis=0)

        #Get the reward with this new weight

        if (per_one == True):
            parameters_to_change_values = list( np.add(np.multiply(best_weight,original_action),original_action) )
            best_actions.append(parameters_to_change_values)
            for s in range(1,size):
                # print("reading data of ",s)
                results[s-1] = comm.recv(source=s,tag=2)
                done[s-1] = comm.recv(source=s,tag=3)
            reward = numpy.average( np.array(results) )
            print("reward")
            print(reward)
            print("\n")
        else:
            parameters_to_change_values = list(best_weight)
            best_actions.append(parameters_to_change_values)
            for s in range(1,size):
                # print("reading data of ",s)
                results[s-1] = comm.recv(source=s,tag=2)
                done[s-1] = comm.recv(source=s,tag=3)
            reward = numpy.average( np.array(results) )
        scores_deque.append(reward)
        scores.append(reward)

        #save the check point
        np.save("Parameters_train_tcp_euc_mocap.npy",np.array(parameters_to_change_values))

        if i_iteration % change_sigma_every == 0:
            sigma = sigma * sigma_reduction_per_one
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=0.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-n_iterations*0.1, np.mean(scores_deque)))
            break

    np.savez("Evolution.npz",scores = np.array(scores),best_parameters = best_actions)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def SlaveProgramCrossEntropyExperimentReward(rank,comm,experiment_number,\
    path_numpy_actions = "Fetch_data/fetch_" ,path_data_excels = "Fetch_data/fetch_",\
    parameters_to_change = ):

    #Experiment assignation
    render_mode = "human"
    #render_mode = "rgb_array"
    render = True
    actions = []
    observations = []
    infos = []

    np_actions_goal_puck = np.load( path_numpy_actions + str(experiment_number[rank]) + '.npz' )
    np_real_robot = env.env.Excel_TCP_Joints_2_Numpy(path_data_excels + str(experiment_number[rank]) + '.xlsx' )
    np_real_robot_pos = np_real_robot[:,:7]
    #Parameters to change loop
    while True:
        parameters_to_change = comm.recv(source=0,tag=1)
        Done = False
        reward = env.env.exp_mocap_tcp_reward(self,parameters_to_change,parameters_values,np_actions_goal_puck,np_real_robot_pos)
        comm.send(reward,dest=0,tag=2)
        Done = True
        comm.send(Done,dest=0,tag=3)






def SlaveProgram(rank,comm):

    start = comm.recv(source=0,tag=1)
    if(start == True):
        print(rank, " has started")
        comm.send(rank,dest=0,tag=2)
        Done = True
        comm.send(Done,dest=0,tag=3)
        print(rank, " has finished")

def SlaveProgram2(rank,env,comm):
    render_mode = "human"
    #render_mode = "rgb_array"
    render = True
    actions = []
    observations = []
    infos = []

    Data_numpy = np.load("Fetch_data/fetch_"+str(rank)+".npz")
    pos_goal = list(Data_numpy["goal"][0,:])
    pos_puck = list(Data_numpy["puck"][0,:])

    env.env.object_pos_from_base = pos_puck[:2]
    env.env.goal_pos_from_base = pos_goal



    actions_numpy = Data_numpy["actions"]
    actions_numpy_gripper = np.zeros([actions_numpy.shape[0],4])
    actions_numpy_gripper[:,:3] = actions_numpy

    while True:
        start = comm.recv(source=0,tag=1)

        if(start == True):
            print(rank, " has started")
            obs = env.reset()
            for i in range (actions_numpy.shape[0]):
                actionRescaled = list(actions_numpy_gripper[i,:])
                if render:
                    env.render(mode=render_mode)
                #time.sleep(0.25)

                obs, reward, done, info = env.step(actionRescaled)
                env.Save_Data_On_Environment()
            env.Reset_Save_Data_On_Environment()

            comm.send(rank,dest=0,tag=2)
            Done = True
            comm.send(Done,dest=0,tag=3)
            print(rank, " has finished")

            start = False
