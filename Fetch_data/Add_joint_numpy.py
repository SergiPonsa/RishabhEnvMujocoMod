import numpy as np
import pandas as pd
#add joint increment
nExp =1
for nExp in range(100,102):
    data = np.load("fetch_"+str(nExp)+".npz")
    DataFrame = pd.read_excel("fetch_"+str(nExp)+".xlsx")
    DFnp = DataFrame.to_numpy()
    list_joint_off = []
    list_joint_off.append( DFnp[0,1:8] - np.array([0, 0.392, 0.0, 1.962, 0.0, 0.78, -1.57]) )
    for i in range(1,DFnp.shape[0]):
        list_joint_off.append(DFnp[i,1:8] - DFnp[i-1,1:8])
    np_joint_off = np.array(list_joint_off)
    #print(np_joint_off)

    np.savez("fetch_"+str(nExp)+".npz",\
        actions_j=np_joint_off,\
        actions=data["actions"],\
        puck=data["puck"],\
        goal=data["goal"],\
        )
