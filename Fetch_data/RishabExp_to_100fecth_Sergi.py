import numpy as np
import pandas as pd
#add joint increment
data = np.load("acs.npy")
data = data[:,:,:3].copy()

nExp =100
for i in range(data.shape[0]):
    np_actions = np.array(data[i,:,:])*0.25
    np_puck = np.ones(np_actions.shape)*[0.5, -0.2, 0.05]
    np_goal = np.ones(np_actions.shape)*[0.6, -0.4, 0.0]
    np.savez("fetch_"+str(nExp+i)+".npz",\
        actions=np_actions,\
        puck=np_puck,\
        goal=np_goal,\
        )
