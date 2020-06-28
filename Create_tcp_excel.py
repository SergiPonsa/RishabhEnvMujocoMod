import pandas as pd
import numpy as np

experiment = 59
origin_robot_respect_world = [0.40023,0.73856,0.3704]
timestep = 10**-3
nsubsteps = 20
FileName="Tcp_Trajectori_"+str(experiment)+".xlsx"
tcps = np.load("Trajectories/tcps.npy")

Dataframe = pd.DataFrame({})

for i in range (3):
    Dataframe["pos " + str(i)] = tcps[experiment,:,i]-origin_robot_respect_world[i]

for i in range (4):
    x = i+3
    Dataframe["quat " + str(i)] = tcps[experiment,:,x]

Dataframe.index = list( np.arange(0,tcps.shape[1]*timestep*nsubsteps,timestep*nsubsteps) )

Dataframe.to_excel(FileName)
