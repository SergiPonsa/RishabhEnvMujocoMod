# RishabhEnvMujocoMod

# This repository is a modified copy of Rishabh Jangir's repository to develop my final Master Thesis.

# Instructions to create and load a conda enviorment, the conda enviorment used in the TFM is saved on Mujoco_Mod_Parameters.yml

'''
Export your environment
conda env export > environment.yml
'''
'''
Load environment
conda env create -f environment.yml 
'''
# Instructions to use the right gym enviorments
'''
cd gym
pip install -e .
'''

## Test Rishabh code

'''
python datagen_sideways_fold.py Gen3SidewaysFold-v0 --mode=demo --render
'''

# Codes developed on Sergi's TFM and utility of them

## Codes to train the Cross Entropy Method using MPI
* function_mpi.py: All the functions used on mpi_hello_word.py and mpi_hello_word_joint.py
* mpi_hello_word.py: This code creates a first master subprocess which assign the different parameters to the simulators, the slave nodes. This program is prepared to train the Cross Entropy Method using the Mocap Control.
* mpi_hello_word_joint.py: This code creates a first master subprocess which assign the different parameters to the simulators, the slave nodes. This program is prepared to train the Cross Entropy Method using the PID Joint Control.
### To run the mpi_hello_world.py or the mpi_hello_world_joint.py the following steps have to be done:
* pip install mpi4py
* mpiexec -n 7 python mpi_hello_world.py 

-n 7 , specifies the number of processors used, in my case I used 1 as master an 6 to simulate, in the following url there is a tutorial to begin using mpi.

https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

## Jupyter notebook to test the enviorment with the modified parameters obtained using Cross Entropy

## Rishabh enviorment modified


## Failed codes on jupyter trying to parallelize the Mujoco simulation
