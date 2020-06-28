# RishabhEnvMujocoMod

Export your environment
conda env export > environment.yml

Load environment
conda env create -f environment.yml

Enviorment Mujoco_Mod_Parameterss

To use the right gym enviorments
cd gym
pip install -e .

Test Rishabh code
python datagen_sideways_fold.py Gen3SidewaysFold-v0 --mode=demo --render
