# RishabhEnvMujocoMod

# This repository is a modified copy of Rishabh Jangir's repository to develop my final Master Thesis.

# Instructions to create and load a conda enviorment, the conda enviorment used in the TFM is saved on Mujoco_Mod_Parameters.yml

Export your environment
conda env export > environment.yml

Load environment
conda env create -f environment.yml 

# Instructions to use the right gym enviorments
cd gym
pip install -e .

Test Rishabh code
python datagen_sideways_fold.py Gen3SidewaysFold-v0 --mode=demo --render

# Codes developed on Sergi's TFM and utility of them
