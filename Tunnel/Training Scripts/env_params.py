"""
This file contains parameters for the execution of grabbing
"""

# Basic libraries needed
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from math import floor
from win32api import GetSystemMetrics
from tqdm import tqdm

# Standard Environment
from Environment import pymunkEnv, calc_JAMoEBA_Radius, Convert, createVideo, save_runtime
import Environment as master_env

# Modified Action and Observation Environment
# from Environment_mod_act_obs import pymunkEnv
# import Environment_mod_act_obs as master_env

# Number of bots in system. 
# This key parameter affects others, so it is kept separate from the rest.
numBots = 10

# Dictionary of how many time steps an episode should last,
# based on how many bots the system is made of
botTimestepDict = {3:40_000,
                    10:3_000,
                    15:6000,
                    20:3000,
                    25:4000,
                    30:2_000}

# Dictionary of pixels-per-meter,
# based on how many bots the system is made of.
botPPMdict = {3:500,
              10:150,
              15:100,
              20:75,
              25:75,
              30:150}

# Environment Parameters    
dt = 1/200.0 # Simulation timestep
numStepsPerStep = 50
botMass = .15
botRadius = .025  
skinRadius = .015 
skinMass = botMass/2
skinRatio = 2
inRadius = botRadius 
botFriction = 0.5
inMass = .0005   
inFriction = botFriction
percentInteriorRemove = 0
springK =  1
springB = 1
springRL = 0 
wallThickness = botRadius/2 
maxSeparation = inRadius*1.75 
energy=False
slidingFriction = 0.2

# Defining system radius
R = calc_JAMoEBA_Radius(skinRadius,skinRatio,botRadius,numBots)

# Target distance from X-start location
targetDistance = R*72 # Unit: m

#Screen parameters (Taken from my big screen (; )
# I.e. use this if operating on any other system
width = 3096
height = 1296

# Esteban's desktop:
width = floor(GetSystemMetrics(0)*.9)
height = floor(GetSystemMetrics(1)*.9)
maxNumSteps = botTimestepDict[numBots]
ppm = botPPMdict[numBots] # Pixels Per Meter

# Parameters for specifiying the system and target locations at start
# If you do not want to specify these, simply set them as 'None'
convert = Convert(botPPMdict[numBots])
render = True
saveVideo = False
dataCollect = False
experimentName = "Testing if Rolling Occurs" # WILL BE OVERWRITTEN IF 'grabbing_RL_params.py' IS CALLED

"""
Items below this comment should not have to be edited by user
"""

# Put all environment changeable parameters into a dictionary. 
envParams = {'dt':dt,
            'numStepsPerStep': numStepsPerStep,
            'ppm':ppm,
            'screenHeight':height,
            'screenWidth':width,
            'maxNumSteps':maxNumSteps,
            'R':R,
            'numBots':numBots,
            'botMass':botMass,
            'botRadius':botRadius,
            'skinRadius':skinRadius,
            'skinMass':skinMass,
            'skinRatio':skinRatio,
            'inRadius':inRadius,
            'botFriction':botFriction,
            'inMass':inMass,
            'inFriction':inFriction,
            'percentInteriorRemove':percentInteriorRemove,
            'springK':springK,
            'springB':springB,
            'springRL':springRL,
            'slidingFriction':slidingFriction,
            'wallThickness':wallThickness,
            'maxSeparation':maxSeparation,
            'dataCollect':dataCollect,
            'experimentName':experimentName,
            'energy':energy,
            'saveVideo':saveVideo
            }