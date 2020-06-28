import pandas as pd
import numpy as np

experiment = 59
timestep = 10**-3
nsubsteps = 20
FileName="Joint_Trajectori_"+str(experiment)+".xlsx"
joints = np.load("Trajectories/joints.npy")

Dataframe = pd.DataFrame({})

for i in range (joints.shape[2]):
    Dataframe["joint " + str(i)] = joints[experiment,:,i]

Dataframe.index = list( np.arange(0,joints.shape[1]*timestep*nsubsteps,timestep*nsubsteps) )

Dataframe.to_excel(FileName)
