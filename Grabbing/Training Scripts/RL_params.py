"""
This file contains all parameters needed for running reinforcememt learning training on the grabbing environmnet
"""

# Import standard libraries
import time
from datetime import date
import os
import sys
import pathlib
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../..')

# Import custom functions
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv

# Loading info from environment needed in this list
from env_params import maxNumSteps, numBots

# Defining Experiment Parameters
experimentNum = '30'
        
# Depending on how many bots there are in the system, 
# the number of training time steps will vary
botTrainingSteps = {#3:1_000_000,
                    # 10:100_000,
                    10:20_000_000,
                    # 15:20_000_000,
                    # 30:30_000_000}
                    # 30:50_000_000}
                    30:10_000_000
                    }


# Training Parameters
neural_network =[1000,1000]
policyName = 'CustomPolicy_'+str(experimentNum)# Name of policy. This can be ignored.
gamma = 0.99                                      # Discount factor
n_steps = 2000                                 # Number of steps to run in each environment per update. Batchsize = n_steps*n_env
ent_coef = 0.00                                # Entropy coefficient
learning_rate = 0.001                          # Learning Rate, can be a funcion
vf_coef = 0.0                                  # Value Function Coefficient in Loss Function
max_grad_norm = 0.5                            # Clipping factor for gradients. Should prevent exploding gradients
lam = 0.95                                     # Factor for bias vs variance for GAE
nminibatches = 8                               # Number of minibatches at each update. Thus the minibatchsize = batchsize//minibatches
noptepochs = 10                                # Number of epochs each update
cliprange = 0.2                                # Cliprange for PPO
seed = 12345                                   # Seed for neural network initialization
nEnvs = 4                                      # Number of parallel environments
training_timesteps = botTrainingSteps[numBots] # number of timesteps to train the environment agent for


# Parameters for callback
check_freq = 50_000 # After how many timesteps do we save the most recent model

# Post training parameters
test=True                       # Whether testing should occur post training
num_tests=3                     # Number of tests with thispolicy to run
render=True                     # Whether to visualize the training
time_per_test = maxNumSteps     # Number of timesteps to run each results episode for.


"""
Users should not have to change anything on below this comment
"""

# Ensuring we are in the proper directory
experimentName = 'Experiment_{}'.format(experimentNum)

# Defining the policy kwargs to be passed
# This makes the neural network that we desire
policy_kwargs = dict(
    net_arch = [dict(
        pi = neural_network,
        vf = neural_network
    )]
)

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback

class CheckpointCallback2(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    A slight modification from the callback above

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback2, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_time_trigger = 0
        self.last_save_name = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        if (self.num_timesteps - self.last_time_trigger) >= self.save_freq:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            if self.last_save_name is None:
                self.last_save_name = path
            else:
                try:
                    os.remove(self.last_save_name +'.zip')
                except:
                    pass
                self.last_save_name = path

            self.model.save(path)
            self.last_time_trigger = self.num_timesteps
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True