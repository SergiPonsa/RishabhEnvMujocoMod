from mpi4py import MPI

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

from functions_mpi import *



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
number_to_experiment = [1,3,5,7,9,11]

if rank == 0:
    print("Hello I am the master rank", str(rank), "of", str(size))
    MasterProgram(size,comm)
else:
    print("Hello I am the slave rank", str(rank), "of", str(size))
    env = envs.make("FetchSlide-v1")
    #SlaveProgram(rank,env)
    SlaveProgram2(rank,env,comm)
