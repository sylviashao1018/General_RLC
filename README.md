# General_RLC
RLC algorithm for general plants
# Installation
Use anaconda to install and manage Python environments:

Download:  
https://www.anaconda.com/

Managing environments info:  
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables

To install Python and all necessary packages:  
`conda env create -f environment.yml`

# RL-Kit
RLKit is the pytorch implementation recommended by the SAC inventors

# MVA_env
Init.m -> Define all parameters und generate code  
model.slx -> Simulink model   
model_ert_shrlib_rtw -> Generated C-Code  
model_win64 -> Binary files  
mva_environment.py -> Model claas   

# main.py

Main file to train SAC on the simulink environment  

# MVA1 

Temperature
Working used for KELI paper

# MVA2 

Temperature and Oxygen 
Clean programmed with vector in and output 
Not finally working

Starting Point for further projects !!!

# MVA3

Temperature
Clean programmed with vector in and output 
Working 

# load policy

## open_rl_agen

load torch policy network and save weights and biases in csv

## importfile

load csv in mathlab

## import_net 

load all policy data and save it in policy struct

## policy_nn

policy function in matlab

# policy

## model.pth

pytorch policy model

## bias_n & weight_n

network parameter

# env_eval 

Policy evaluations environment 








