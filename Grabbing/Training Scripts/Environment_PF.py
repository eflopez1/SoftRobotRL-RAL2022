import pymunk
import pygame
import sys
import numpy as np
from numpy import sin, cos, sqrt, log
from numpy.linalg import norm
from math import floor
from gym import spaces, Env
import matplotlib.pyplot as plt
from datetime import datetime
from warnings import warn
import os
import pdb

class pymunkEnv(Env):
    
    info = {'There is no info to be aware of.':'Machinelearn away.'}
    
    def __init__(self, dt, ppm, screenHeight, screenWidth, 
                maxNumSteps, R, numBots, botMass, botRadius, 
                skinRadius, skinMass, skinRatio, inRadius, 
                botFriction, inMass, inFriction, percentInteriorRemove, 
                springK, springB, springRL, wallThickness, maxSeparation, 
                binding_spring_K,
                dataCollect=False, experimentName="NOT NAMED", saveVideo = False, 
                energy=False, velocityPenalty=False, numStepsPerStep=1,
                slidingFriction=0):
        """
        All units in this function should be in standard physical units:
            Distance: Meters
            Force: N
            Velocity: m/s
            Mass: kg
        """

        self.report_all_data = False # If True, will report data points for ALL items in simulation, including obstacles and walls.

        # Basic simulation parameters
        self.dt = dt                                                 # Simulation timestep
        self.ppm = ppm                                               # Pixels per Meter
        self.convert = Convert(ppm)                                  # Conversion to be used for all parameters given
        self.height= self.convert.Pixels2Meters(screenHeight)        # Height of the screen, in pixels
        self.width = self.convert.Pixels2Meters(screenWidth)         # Width of the screen, in pixels
        self.maxNumSteps = maxNumSteps                               # Number of steps until simulation terminated
        self.maxVelocity = 20                                        # Arbitrarily set, may need changing later.
        self.forceGain = 2.0                                   # Defining the maximum force a bot can apply in any given direction
        self.dataCollect = dataCollect                               # Are we collecting data rn?
        self.saveVideo = saveVideo
        self.experimentName = experimentName                         # Experiment name. Is assigned to plots folder and video
        self.energy=energy                                           # If True, then we do care about calculating how much energy our system is expending to complete its mission
        self.velocityPenalty = velocityPenalty
        self.numStepsPerStep = numStepsPerStep                       # The number of simulation timesteps to run for each call to 'step' function
        self.slidingFriction = slidingFriction                       # Coefficient of friction for objects sliding on the ground
        
        # If we are penalizing the non moving box, create a horizon
        if self.velocityPenalty:
            self.velRecent = np.zeros(250) # We will store 30 timesteps worth of information

        # JAMoEBA system membrane parameters
        self.R = R
        self.numBots = numBots
        self.botMass = botMass
        self.botRadius = botRadius     # Radius of 
        self.botFriction = botFriction # Friction of bots and skins
        self.skinRadius = skinRadius
        self.skinMass = skinMass
        self.skinRatio = skinRatio
        
        # Spring parameters
        self.springK = springK
        self.springB = springB
        self.springRL = springRL
        self.maxSeparation = maxSeparation
        
        # Interior parameters
        self.inRadius = inRadius
        self.inMass = inMass
        self.inFriction = inFriction
        self.percentInteriorRemove = percentInteriorRemove
        
        # Thickness of wall
        self.wallThickness = wallThickness 

        # Square size
        self.squareLength = self.R * .75

        # Object Starting Position
        objectX = wallThickness*3 + 2.1*(R + botRadius*2) # Start on left side of environment
        objectY = self.height/2
        self.objectPosition = np.array([objectX, objectY])

        # Strength of spring holding square in-place
        self.binding_spring_K = binding_spring_K
        self.binding_springs_removed = False # Will change to true if the springs have been removed

        # Start location of JAMoEBA system
        self.start_options = [
            [wallThickness*3 + R + botRadius*2, objectY] # Left side of screen
        ]
        self.num_starts = len(self.start_options)
        whichStart = np.random.randint(0,self.num_starts)
        self.systemStart = self.start_options[whichStart]

        # Location to move object to
        self.targetLoc = np.array([self.width/2, self.height/2])
        self.targetDistance = np.linalg.norm(self.targetLoc - np.asarray(self.objectPosition)) # Defining the initial distance to target
            
        #### Gym API
        self.state_size = self.numBots*5 + 3
        self.action_size = 2*self.numBots
        force_low, force_high = -1.0, 1.0 # Defining how large (and in each direction) a force may be applied by the agent

        low = -10
        high = 10
        self.observation_space = spaces.Box(low, high, 
                                            shape=(self.state_size,), # [Posx, Posy, Velx, Vely, Ax, Ay, Fx, Fy], not including square information
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=force_low, 
                                        high=force_high, 
                                        shape=(self.action_size,),
                                        dtype=np.float32)
        
        #Gather information on number of interior
        granPerRing, _ = interiorPattern(self.R, self.inRadius, self.botRadius, self.percentInteriorRemove)
        self.numInterior = np.sum(granPerRing)
        
        #### Data Collection
        self.environment_parameters = [
                                        ['dt:',str(self.dt)],
                                       ['NumSetpsPerStep:',str(self.numStepsPerStep)],
                                       ['Num_Bots', str(self.numBots)],
                                       ['botMass',str(self.botMass)],
                                       ['botRadius',str(self.botRadius)],
                                       ['skinRadius',str(self.skinRadius)],
                                       ['skinMass',str(self.skinMass)],
                                       ['skinRatio',str(self.skinRatio)],
                                       ['inRadius',str(self.inRadius)],
                                       ['botFriction',str(self.botFriction)],
                                       ['inMass',str(self.inMass)],
                                       ['inFriction',str(self.inFriction)],
                                       ['percentInteriorRemoved',str(self.percentInteriorRemove)],
                                       ['SpringK:', str(self.springK)], 
                                       ['SpringB:', str(self.springB)], 
                                       ['SpringRL:', str(self.springRL)],
                                       ['Binding_Spring_K', str(self.binding_spring_K)],
                                       ['maxSeparation',str(self.maxSeparation)],
                                       ['slidingFriction', str(self.slidingFriction)],
                                       ['energy',str(energy)],
                                       ['velocityPenalty',str(self.velocityPenalty)],
                                       ['PixelsPerMeter (ppm):',str(self.ppm)],
                                       ['SystemRadius',str(self.R)],
                                       ['ScreenWidth',str(self.width)],
                                       ['ScreenHeight',str(self.height)],
                                       ['Num_Interior:', str(self.numInterior)],
                                       ['Bot_Radius:', str(self.botRadius)],
                                       ['JAMoEBA_Radius:', str(self.R)], 
                                       ['Square Length:', str(self.squareLength)],
                                       ['maxNumSteps',str(self.maxNumSteps)]
                                       ]
        
        if self.dataCollect:
            now = str(datetime.now())
            now = now.replace(":","")
            now = now[:-7]
            
            self.saveFolder = experimentName+ " Data and Plots " + now + "/"
            os.makedirs(self.saveFolder,exist_ok=True)
            # This +1 is for the extra column needed to record time.
            self.X_data = np.zeros(self.numBots + 1)     # X-Position of each bot
            self.X_vel_data = np.zeros(self.numBots + 1) # X-Velocity of each bot
            self.Y_data = np.zeros(self.numBots + 1)     # Y-Position of each bot
            self.Y_vel_data = np.zeros(self.numBots + 1) # Y-Velocity of each bot
            self.ac = np.zeros(self.action_size + 1)     # Action taken on each bot
            self.reward_data = np.zeros(2)               # Reward signal sent to RL
            self.obs_data = np.zeros(self.state_size +1) # Observation at each time step
            self.objPos = np.zeros(3)                    # Position of box
            self.objVel = np.zeros(3)                    # Obj velocity
            self.objOmega = np.zeros(2)                  # Obj angular velocity
        
        if self.saveVideo:
            self.videoFolder = experimentName + '_VideoImages/'
            os.makedirs(self.videoFolder,exist_ok=True)

        return None
        
        
        
        
        
    def reset(self, newStartingLoc=None):
        """
        Users now have the option to change the starting location of the system when it is spawned
        """ 
        
        self.render_setup = False # Changes to true once the rendering tools have been setup. This will only happen externally if rendering has been requested
        self.space = pymunk.Space()
        self.space.gravity = 0,0
        self.timestep = 0 # Initializing timestep
        self.time = 0     # Initializing time

        # For feeding NN the last action taken
        self.last_action = np.zeros(self.numBots*2)
        
        # Information for contact, will be changed later
        self.extForcesX = np.zeros(self.numBots)
        self.extForcesY = np.zeros(self.numBots)
        self.botContacts = np.zeros(self.numBots)

        
        # Converting units to pixel coordinates before feeding into space for creation
        R = self.convert.Meters2Pixels(self.R)
        systemStart = self.convert.Meters2Pixels(self.systemStart[0]), self.convert.Meters2Pixels(self.systemStart[1])
        botRadius = self.convert.Meters2Pixels(self.botRadius)
        skinRadius = self.convert.Meters2Pixels(self.skinRadius)
        inRadius = self.convert.Meters2Pixels(self.inRadius)
        springK = self.convert.SpringK2Pixels(self.springK)
        springRL = self.convert.Meters2Pixels(self.springRL)
        maxSeparation = self.convert.Meters2Pixels(self.maxSeparation)
        length = self.convert.Meters2Pixels(self.squareLength)
        bind_spring_K = self.convert.SpringK2Pixels(self.binding_spring_K)
        objectPosition = self.convert.Meters2Pixels(self.objectPosition)
        
        height = self.convert.Meters2Pixels(self.height)
        width = self.convert.Meters2Pixels(self.width)
        wallThickness = self.convert.Meters2Pixels(self.wallThickness)
        
        # Updating the system start for any future resets
        whichStart = np.random.randint(0, self.num_starts)
        self.systemStart = self.start_options[whichStart]

        if newStartingLoc is not None:
            systemStart = self.convert.Meters2Pixels(newStartingLoc[0]), self.convert.Meters2Pixels(newStartingLoc[1])

        #### Create the system
        # All items that require conversion has been converted above
        kwargs={'space':self.space,
            'systemCenterLocation':systemStart,
            'systemRadius':R,
            'numBots':self.numBots,
            'botMass':self.botMass,
            'botRadius':botRadius,
            'skinMass':self.skinMass,
            'skinRadius':skinRadius,
            'skinRatio':self.skinRatio,
            'botFriction':self.botFriction,
            'inRadius':inRadius,
            'inMass':self.inMass,
            'inFriction':self.inFriction,
            'springK':springK,
            'springB':self.springB,
            'springRL':springRL,
            'maxSeparation':maxSeparation,
            'percentInteriorRemove':self.percentInteriorRemove,
            'botCollisionIntStart': 2} # All obstacles will be of collision_type=1
        self.bots, self.skinParticles, self.interiorParticles = createJamoeba(**kwargs)
        
        # Note: Membrane consists of bots and skin particles
        self.jamoeba = [*self.bots, *self.skinParticles, *self.interiorParticles] # This is all bodies within the space! 
        
        #### Create Walls
        wallKwargs={'space':self.space,
                    'screenHeight':height,
                    'screenWidth':width,
                    'wallThickness':wallThickness,
                    'env':self}
        _, _, _, wall4 = createWalls(**wallKwargs) 
        _, wall4_body = wall4
        self.wall_loc = self.convert.Pixels2Meters(np.asarray(wall4_body.position))
        
        #### Add object to grab
        objectMass = 0.5*self.botMass*self.numBots + self.skinMass*(self.skinRatio*self.numBots) + self.inMass*self.numInterior
        objectKwargs = dict(
            space=self.space,
            position=tuple(objectPosition),
            length=length,
            friction=self.botFriction,
            mass=objectMass,
            rotation=np.pi/4
        )
        self.square = squareObject(**objectKwargs)        

        # Add slot to get square into
        slotKwargs = dict(
            space = self.space,
            squareLength = length,
            friction = self.botFriction,
            squarePosition = tuple(objectPosition),
            wallThickness=wallThickness,
            report_slot = self.report_all_data
        )
        if self.report_all_data:
            targetLoc, self.slot_points =  createSlot(**slotKwargs)
            self.slot_points = self.convert.Pixels2Meters(self.slot_points)
        else:
            targetLoc = createSlot(**slotKwargs)
        targetLoc = np.asarray(targetLoc)
        self.targetLoc = self.convert.Pixels2Meters(targetLoc)


        # Store all objects in a single list to add friction
        if not self.square.static:
            self.all_objects = [self.square, *self.jamoeba]
        else:
            self.all_objects = [*self.jamoeba]

        if self.slidingFriction > 0:
            add_sliding_friction(self.space, self.all_objects, self.slidingFriction)

        # Creating the pot field
        pot_field = create_pot_field(self)
        self.control = controller(pot_field)

        #### Collision Handler
        # Reports collisions with walls, objects, and obstacles
        for bot in self.bots:
            cHandler = self.space.add_collision_handler(1,bot.shape.collision_type)
            cHandler.post_solve = self.colPost
        
        if self.report_all_data:
            self.botPositions = []
            self.skinPositions = []
            self.interiorPositions = []
            self.movingObjectPosition = []

        #### Initiate Observation
        observation, _, self.previousDistance, _ = self.getOb()
        return observation
    
    
    
        
    def step(self, ac):
        
        # Contact information for storing
        self.extForcesX = np.zeros(self.numBots)
        self.extForcesY = np.zeros(self.numBots)
        self.botContacts = np.zeros(self.numBots)

        # Define forces for this step
        forcesX = []
        forcesY = []
        if self.dataCollect:
            forcesApplied = []

        for index in range(self.numBots):
            
            xForce = ac[2*index]*self.forceGain
            yForce = ac[2*index+1]*self.forceGain

            forcesX.append(xForce)
            forcesY.append(yForce)

            if self.dataCollect:
                forcesApplied.append(xForce)
                forcesApplied.append(yForce)
            
        forcesX_ANN = np.asarray(forcesX, dtype=np.float) # Forces as given by the neural network
        forcesY_ANN = np.asarray(forcesY, dtype=np.float) # Forces as given by the neural network
                    
        # # Get the pot field forces as well
        potFx, potFy = self.control.get_action()
        forcesX = forcesX_ANN + potFx*self.forceGain
        forcesY = forcesY_ANN + potFy*self.forceGain

        # Apply those forces for the number of steps-per-step
        for i in range(self.numStepsPerStep):
            for index, bot in enumerate(self.bots):
                xForce = forcesX[index]
                yForce = forcesY[index]
                
                botPos = bot.body.position
                bot.body.apply_force_at_world_point((xForce, yForce), (botPos.x, botPos.y))

        # Taking a step in the environment
            self.space.step(self.dt)
            self.time += self.dt

        # Note that we are considering timestep and time as different!
        self.timestep+=1
        
        # Gather information
        obs, systemCenter, distanceToTarget, distanceToSquare = self.getOb()
        self.last_action = ac
            
        rew = self.calcRew(distanceToTarget, distanceToSquare)
        isDone, rew = self.isDone(rew, systemCenter, distanceToTarget)
        self.previousDistance = distanceToTarget
        
        if self.dataCollect: 
            self.dataCollection(forcesApplied,rew,obs)

        if self.report_all_data:
            self.collectAllData() # If we are reporting all data, then we must report all data positions
 
        return obs, rew, isDone, self.info
    
    
    def getOb(self):
        
        runTime = [self.timestep/self.maxNumSteps] # Can use if you want to feed the system information on how much time is left
        
        observation = np.empty((self.numBots,8))

        # Getting the square's data first, since that does not change.
        squarePos = self.convert.Pixels2Meters(self.square.body.position.x), self.convert.Pixels2Meters(self.square.body.position.y)
        squareVel = self.convert.Pixels2Meters(self.square.body.velocity.x), self.convert.Pixels2Meters(self.square.body.velocity.y)
        squareTheta = self.square.body.angle / np.pi                       # Normalized rotation of square
        squareOmega = self.square.body.angular_velocity / self.maxVelocity # Normalized angular velocity

        botPos = np.zeros((self.numBots,2))
        botVel = np.zeros((self.numBots,2))
        for index, bot in enumerate(self.bots):
            currentBotPos = self.convert.Pixels2Meters(bot.body.position.x), self.convert.Pixels2Meters(bot.body.position.y)
            currentBotVel = self.convert.Pixels2Meters(bot.body.velocity.x), self.convert.Pixels2Meters(bot.body.velocity.y)

            botPos[index,:] = np.array(currentBotPos) - np.array(squarePos)
                
            botVel[index,:] = currentBotVel
        
        squareLocation = np.array(squarePos) - np.array(self.targetLoc)

        # This value is in meters. Distance of square to target location
        distanceToTarget = norm(squareLocation) 

        # Calculating the distance to 
        distanceToSquare = norm(np.mean(botPos,axis=0))

        # Normalizing observation
        botPos[:,0]=botPos[:,0]/(self.width)                   #Normalizing X-Coordinate
        botPos[:,1]=botPos[:,1]/(self.height)                  #Normalizing Y-Coordinate
        botVel /= self.maxVelocity                             #Normalizing Velocity

        # Normalizing square data
        squareVel = np.asarray(squareVel) / self.maxVelocity
        
        # Creating an observation
        observation = np.concatenate((squareLocation, [squareTheta], botPos.flatten(), botVel.flatten(), self.botContacts))
        
        return observation, squareLocation, distanceToTarget, distanceToSquare


    def reportContact(self, contactPair, impulse):
        botIndex = max(contactPair)
        forces = impulse/self.dt
        # Doing a += so that as multiple contacts occur over the many timesteps, we add them
        self.extForcesX[botIndex-2] = forces[0]
        self.extForcesY[botIndex-2] = forces[1]
        self.botContacts[botIndex-2] = 1
    
    
    def colPost(self, arbiter, space, data):
        impulse = arbiter.total_impulse
        collisionShapes = arbiter.shapes
        collisionPair = [collisionShapes[0].collision_type, collisionShapes[1].collision_type]
        self.reportContact(collisionPair, impulse)
        return True
    
    
    def calcRew(self, distanceToTarget, distanceToSquare):
        rew = 0
        
        progress = self.previousDistance - distanceToTarget # Relative to the velocity or speed for arriving at target
        rew += progress*400

        # Let's give a small encouragement to twist the block. This may help get it into place.
        omega = np.abs(self.square.body.angular_velocity)
        rew += omega*10

        return rew
    
    
    def isDone(self, rew, systemCenter, distanceToTarget):
        done=False
        # Takes too long to complete
        if self.timestep>self.maxNumSteps:
            done=True
            rew -= 1 

        if distanceToTarget < self.R/10:
            rew += 10
            done=True

        return done, rew
            
    
    def render(self):
        targetLoc = self.convert.Meters2Pixels(self.targetLoc)
        if not self.render_setup:
            from pymunk.pygame_util import DrawOptions
            
            screenHeight = floor(self.convert.Meters2Pixels(self.height))
            screenWidth = floor(self.convert.Meters2Pixels(self.width))
            
            pygame.init()
            self.screen = pygame.display.set_mode((screenWidth, screenHeight))
            self.clock = pygame.time.Clock()
            
            """
            Review this link for information on drawing in PyGame
            http://www.pymunk.org/en/latest/pymunk.html#pymunk.SpaceDebugDrawOptions
            """
            self.drawOptions = DrawOptions(self.screen)
            self.drawOptions.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES 
            self.drawOptions.shape_outline_color = (0,0,0,255)
            
            pymunk.pygame_util.positive_y_is_up=True
            # Adding a visual for the target
            targetLoc = pymunk.pygame_util.to_pygame(targetLoc, self.screen) # Converting to pygame coordinates
            pygame.draw.circle(self.screen, (255,0,0), targetLoc, radius = 5)
            
            self.render_setup=True
            
        for event in pygame.event.get():
            if event.type==pygame.QUIT: 
                pygame.display.quit()
                self.close()
            if event.type==pygame.KEYDOWN and event.key == pygame.K_ESCAPE: 
                pygame.display.quit()
                self.close()
        
        
        self.screen.fill((192,192,192))
        self.space.debug_draw(self.drawOptions)
        # Adding a visual for the target
        targetLoc = pymunk.pygame_util.to_pygame(targetLoc, self.screen) # Converting to pygame coordinates
        pygame.draw.circle(self.screen, (255,0,0), targetLoc, radius = 5)

        pygame.display.update()
        if self.saveVideo:# and self.timestep%10==0:
            pygame.image.save(self.screen, self.videoFolder+'image%06d.jpg' % self.timestep)
        self.clock.tick()

    def close(self):
        if self.render_setup:
            pygame.display.quit()
        del self.space
            
            
    def dataCollection(self,forcesApplied,rew,obs):
        
        # Create new and empty vecotrs
        X_Pos_temp = [self.time]
        Y_Pos_temp = [self.time]
        
        X_vel_temp = [self.time]
        Y_vel_temp = [self.time]
        
        action_temp = [self.time]
        rew_temp = [self.time]
        obs_temp = [self.time]
        
        for bot in self.bots:
            currentBotPos = self.convert.Pixels2Meters(bot.body.position.x)-self.targetLoc[0], self.convert.Pixels2Meters(bot.body.position.y)-self.targetLoc[1]
            currentBotVel = self.convert.Pixels2Meters(bot.body.velocity.x), self.convert.Pixels2Meters(bot.body.velocity.y)
            
            X_Pos_temp.append(currentBotPos[0])
            Y_Pos_temp.append(currentBotPos[1])
            
            X_vel_temp.append(currentBotVel[0])
            Y_vel_temp.append(currentBotVel[1])
            
            
        for action in forcesApplied:
            action_temp.append(action)
        
        rew_temp.append(rew)
            
        for observation in obs.flatten():
            obs_temp.append(observation)

        # Convert to Numpy Arrays
        X_Pos_temp = np.asarray(X_Pos_temp)
        Y_Pos_temp = np.asarray(Y_Pos_temp)
        
        X_vel_temp = np.asarray(X_vel_temp)
        Y_vel_temp = np.asarray(Y_vel_temp)
        
        action_temp = np.asarray(action_temp)
        rew_temp = np.asarray(rew_temp)
        obs_temp = np.asarray(obs_temp)
        
        # Now append to the master list
        # Refer to __init__ method to learn what each of these variables are
        self.X_data = np.vstack([self.X_data, X_Pos_temp])
        self.X_vel_data = np.vstack([self.X_vel_data, X_vel_temp])
        self.Y_data = np.vstack([self.Y_data, Y_Pos_temp])
        self.Y_vel_data = np.vstack([self.Y_vel_data, Y_vel_temp])
        self.ac = np.vstack([self.ac, action_temp])
        self.reward_data = np.vstack([self.reward_data, rew_temp])
        self.obs_data = np.vstack([self.obs_data, obs_temp])

        # Object information
        objPosTemp = [self.time]
        objVelTemp = [self.time]
        objOmegaTemp = [self.time]
        for index, pos in enumerate(self.square.body.position):
            objPosTemp.append(self.convert.Pixels2Meters(pos) - self.targetLoc[index])
        for vel in self.square.body.velocity:
            objVelTemp.append(self.convert.Pixels2Meters(vel))
        omega = self.square.body.angular_velocity
        objOmegaTemp.append(omega)
        
        self.objPos = np.vstack([self.objPos, np.asarray(objPosTemp)])
        self.objVel = np.vstack([self.objVel, np.asarray(objVelTemp)])
        self.objOmega = np.vstack([self.objOmega, np.asarray(objOmegaTemp)])
            
        
    def collectAllData(self):
        """
        Will collect all data and store in the global list made in __init__ method
        """
        tempBotPositions = [self.time]
        tempSkinPositions = [self.time]
        tempInteriorPositions = [self.time]
        tempMovingObjectPosition = [self.time]

        # Get all positions
        for bot in self.bots:
            botPos = np.asarray(bot.body.position)
            botPos = self.convert.Pixels2Meters(botPos)
            tempBotPositions.append(botPos)
        
        for skin in self.skinParticles:
            skinPos = np.asarray(skin.body.position)
            skinPos = self.convert.Pixels2Meters(skinPos)
            tempSkinPositions.append(skinPos)

        for interior in self.interiorParticles:
            inPos = np.asarray(interior.body.position)
            inPos = self.convert.Pixels2Meters(inPos)
            tempInteriorPositions.append(inPos)

        movingObjectPosition = np.asarray(self.square.body.position)
        movingObjectPosition = self.convert.Pixels2Meters(movingObjectPosition)
        tempMovingObjectPosition.append(movingObjectPosition)

        # Finally, get the orientation of the moving object, as that is not currently grabbed
        tempMovingObjectPosition.append(self.square.body.angle)

        # Unwrap all lists
        tempBotPositions = list(flatten(tempBotPositions))
        tempSkinPositions = list(flatten(tempSkinPositions))
        tempInteriorPositions = list(flatten((tempInteriorPositions)))
        tempMovingObjectPosition = list(flatten(tempMovingObjectPosition))

        # Append to the master lists
        self.botPositions.append(tempBotPositions)
        self.skinPositions.append(tempSkinPositions)
        self.interiorPositions.append(tempInteriorPositions)
        self.movingObjectPosition.append(tempMovingObjectPosition)





    def parameterExport(self, saveLoc=None):
        if saveLoc==None:
            parameter_file=  self.experimentName + '_Environment_parameters.txt'
            print('\n','--'*20)
            warn('No Save location for environment parameters has been specificed')
            print('--'*20,'\n')
        else:
            os.makedirs(saveLoc,exist_ok=True)
            parameter_file = saveLoc+'Environment_parameters.txt'
        
        with open(parameter_file, 'w') as f:
            for line in self.environment_parameters:
                f.write("%s\n" % line)
            f.write('\n')
            # TODO: Update this to be included only when training is occuring
            # f.write("Number of Training Episodes:{}".format(str(self.episode)))
            
            
    def dataExport(self):
        
        print('\n\nExporting and plotting data...',end='')

        # Delete the temporarily made first row of zeroes
        self.X_data = np.delete(self.X_data, 0, 0)
        self.X_vel_data = np.delete(self.X_vel_data, 0, 0)
        self.Y_data = np.delete(self.Y_data, 0, 0)
        self.Y_vel_data = np.delete(self.Y_vel_data, 0, 0)
        self.ac = np.delete(self.ac, 0, 0)
        self.reward_data = np.delete(self.reward_data, 0, 0)
        self.obs_data = np.delete(self.obs_data, 0, 0)
        self.objPos = np.delete(self.objPos, 0, 0)
        self.objVel = np.delete(self.objVel, 0, 0)
        self.objOmega = np.delete(self.objOmega, 0, 0)
        

        # Save the data on .csv files
        np.savetxt(self.saveFolder + 'X_data.csv', self.X_data, delimiter=',')
        np.savetxt(self.saveFolder + 'X_vel_data.csv', self.X_vel_data, delimiter=',')
        np.savetxt(self.saveFolder + 'Y_data.csv', self.Y_data, delimiter=',')
        np.savetxt(self.saveFolder + 'Y_vel_data.csv', self.Y_vel_data, delimiter=',')
        np.savetxt(self.saveFolder + 'actions.csv', self.ac, delimiter=',')
        np.savetxt(self.saveFolder + 'reward.csv', self.reward_data, delimiter=',')
        np.savetxt(self.saveFolder + 'observations.csv', self.obs_data, delimiter=',')
        np.savetxt(self.saveFolder + 'object_position.csv', self.objPos, delimiter=',')
        np.savetxt(self.saveFolder + 'object_velocity.csv', self.objVel, delimiter=',')
        np.savetxt(self.saveFolder + 'object_omega.csv', self.objOmega, delimiter=',')
        
        self.plot_data()
        self.parameterExport(self.saveFolder)
        
        print('Data Export and Plot Complete')



    def plot_data(self):
        # Common Among Position and Velocity Data
        last_col = len(self.X_data[0])-1
        time = self.X_data[:,0]
        xlabel = 'Time [sec]'
        
        # Plot X-Position
        X_COM = []
        for row in self.X_data:
            pos = np.mean(row[1:last_col])
            X_COM.append(pos)
        plt.figure('X-Pos')
        plt.plot(time,X_COM)
        plt.xlabel(xlabel)
        plt.ylabel('X-Position (rel to target) [m]')
        plt.title('X-Center Position')
        plt.savefig(self.saveFolder + 'X-Center Position.jpg')       
        
        # Plot Y-Position
        Y_COM = []
        for row in self.Y_data:
            pos = np.mean(row[1:last_col])
            Y_COM.append(pos)
        plt.figure('Y-Pos')
        plt.plot(time,Y_COM)
        plt.xlabel(xlabel)
        plt.ylabel('Y-Position (rel to target) [m]')
        plt.title('Y-Center Position')
        plt.savefig(self.saveFolder + 'Y-Center Position.jpg')
        
        # Plot X-velocity
        plt.figure('X-Vel')
        for i in range(self.numBots):
            plt.plot(time, self.X_vel_data[:,i+1], label = 'Bot' + str(i+1))
        plt.xlabel(xlabel)
        plt.ylabel('X Velocity [m/s]')
        plt.title('X-Velocity')
        plt.legend(loc='lower right')
        plt.savefig(self.saveFolder + 'X-Velocity.jpg')
        
        # Plot Y-Velocity
        plt.figure('Y-Vel')
        for i in range(self.numBots):
            plt.plot(time, self.Y_vel_data[:,i+1], label = 'Bot' + str(i+1))
        plt.xlabel(xlabel)
        plt.ylabel('Y Velocity [m/s]')
        plt.title('Y-Velocity')
        plt.legend(loc='lower right')
        plt.savefig(self.saveFolder + 'Y-Velocity.jpg')        
        
        # Plot object position
        plt.figure('ObjectPosition')
        plt.plot(time, self.objPos[:,1],label='X-Pos')
        plt.plot(time, self.objPos[:,2],label='Y-Pos')
        plt.xlabel(xlabel)
        plt.ylabel('Position (rel to target) [m]')
        plt.title('Object Position')
        plt.legend(loc='best')
        plt.savefig(self.saveFolder+'Object-Position,jpg')

        # Plot object velocity
        plt.figure('ObjectVelocity')
        plt.plot(time,self.objVel[:,1],label='X-Vel')
        plt.plot(time,self.objVel[:,2],label='Y-Vel')
        plt.xlabel(xlabel)
        plt.ylabel('Velocity [m/s]')
        plt.title('Object Velocity')
        plt.legend(loc='best')
        plt.savefig(self.saveFolder+'ObjectVelocity.jpg')

        # Plot object omega
        plt.figure('ObjectOmega')
        plt.plot(time, self.objOmega[:,1])
        plt.xlabel(xlabel)
        plt.ylabel('Omega [rad/s]')
        plt.title('Object Angular Velocity')
        plt.savefig(self.saveFolder+'ObjectOmega.jpg')

        # Plot Actions
        last_col_2 = len(self.ac[0])-1
        bot=1
        for i in range(last_col_2):
            if i%2!=0:
                plt.figure('Applied Forces Bot ' + str(bot))
                plt.plot(time, self.ac[:,i], label='X-Force')
                plt.plot(time, self.ac[:,i+1], label='Y-Force')
                plt.xlabel(xlabel)
                plt.ylabel('Force [N]')
                plt.title('Applied Forces on Bot ' + str(bot))
                plt.legend(loc='lower right')
                plt.savefig(self.saveFolder + 'Bot ' + str(bot) + ' Applied Forces.jpg')
                bot+=1
                
        # Plot reward
        time = self.reward_data[:,0]
        rewards = self.reward_data[:,1]
        plt.figure('Rewards')
        plt.plot(time, rewards)
        plt.xlabel(xlabel)
        plt.ylabel('Reward')
        plt.title('Reward for JAMoEBA')
        plt.savefig(self.saveFolder + 'Reward.jpg')
        
            

    def exportAllData(self):
        """
        Will export all data that was collected 
        as a result of self.report_all_data
        """
        np.save(self.saveFolder + 'bot_coords', np.asarray(self.botPositions))
        np.save(self.saveFolder + 'skin_coords', np.asarray(self.skinPositions))
        np.save(self.saveFolder + 'interior_coords', np.asarray(self.interiorPositions))
        np.save(self.saveFolder + 'target_loc', np.asarray(self.targetLoc))
        np.save(self.saveFolder + 'square_loc', np.asarray(self.movingObjectPosition))
        try:
            np.save(self.saveFolder + 'slot Points', np.asarray(self.slot_points))
        except:
            print('Unable to save slot points')

    
class Convert:
    def __init__(self, conversion_ratio):
        """
        Parameters
        ----------
        conversion_ratio : float
            The conversion ratio between pixels and meters in the form (pixels/meter).
        """
        self.ratio = conversion_ratio
        
    def Pixels2Meters(self, num_pixels):
        return (num_pixels*(1/self.ratio))
    
    def Meters2Pixels(self, meters):
        return (meters*self.ratio)
    
    def SpringK2Pixels(self, springK):
        return (springK*(1/self.ratio))
    
    def Pixels2SpringK(self, springKPixels):
        return (springKPixels*self.ratio)

        
class squareObject:
    def __init__(self, space, position, mass, length, friction, rotation=0, color = (0,0,0,255), static=False):
        self.static = static
        if static:
            self.body = pymunk.Body(0,0, body_type=pymunk.Body.STATIC) # Static Body
        else:
            self.body = pymunk.Body() # Kinematic body

        self.length = length
        self.shape = pymunk.Poly.create_box(self.body, (length,length))
        self.shape.body.position = position
        self.shape.body.angle = rotation
        self.shape.friction = friction
        if not static:
            self.shape.mass = mass # Only moving bodies can have a mass
        self.shape.collision_type = 1

        # Add to space
        space.add(self.body, self.shape)

def connectBalls(space, theta1, theta2, b1, b2, rest_length, spring_stiffness, spring_damping, maxSeparation):
    springConstraint = pymunk.DampedSpring(b1.body, b2.body, 
                                           (-b1.radius*np.sin(theta1), b1.radius*np.cos(theta1)), (b2.radius*np.sin(theta2), -b2.radius*np.cos(theta2)), 
                                            rest_length, spring_stiffness, spring_damping)
    slideJoint = pymunk.SlideJoint(b1.body, b2.body, 
                                   (-b1.radius*np.sin(theta1), b1.radius*np.cos(theta1)), (b2.radius*np.sin(theta2), -b2.radius*np.cos(theta2)), 
                                   0, maxSeparation)
    space.add(springConstraint, slideJoint)
    return None

def createJamoeba(space, systemCenterLocation, systemRadius, numBots, botMass, botRadius, skinMass, skinRadius, skinRatio, botFriction, springK, springB, springRL, maxSeparation, inRadius, inMass, inFriction,  percentInteriorRemove = 0, botCollisionIntStart = 2):
    xCenter = systemCenterLocation[0]
    yCenter = systemCenterLocation[1]
    
    collisionType = botCollisionIntStart      
    
    bots = []
    interiorParticles = []
    membrane = []
    skinParticles = []
    
    bots_only = False

    # Get interior particles
    ## Comment out if you only want bots
    if not bots_only:
        gran_per_ring, in_rings_radius = interiorPattern(systemRadius, inRadius, botRadius, percentInteriorRemove)
    
    #Parameter for skins
    t = (2*np.pi/numBots)/(skinRatio+1)
    
    for i in range(numBots):
        theta = i*2*np.pi/numBots
        x = xCenter + systemRadius*np.cos(theta)
        y = yCenter + systemRadius*np.sin(theta)
        
        bot = Ball(space, (x,y), botRadius, botMass, botFriction, collisionType, color=(255,0,0,255))
        collisionType += 1
        bots.append(bot)
        membrane.append(bot)
        
        # Skin particles
        # Comment out if you only want bots
        if not bots_only:
            for j in range(1,skinRatio+1):
                x = xCenter + systemRadius*np.cos(theta + j*t)
                y = yCenter + systemRadius*np.sin(theta + j*t)
                skin = Ball(space, (x,y), skinRadius, skinMass, botFriction, color=(0,0,255,255))
                membrane.append(skin)
                skinParticles.append(skin)

    # # Comment out if you only want bots  
    if not bots_only:  
        numBodies = len(membrane)
        for index, body in enumerate(membrane):
            if index < (numBodies-1):
                connectBalls(space, t*index, t*(index+1), body, membrane[index+1], springRL, springK, springB, maxSeparation)
            else:
                connectBalls(space, t*index, 0, body, membrane[0], springRL, springK, springB, maxSeparation)
    
    # Create Interiors
    # # Comment out if you only want bots
    if not bots_only:
        for index, in_ring in enumerate(gran_per_ring):
            radius = in_rings_radius[index]
            for j in range(in_ring):
                in_theta = j*2*np.pi/in_ring
                x = xCenter + radius*np.cos(in_theta)
                y = yCenter + radius*np.sin(in_theta)
                
                interiorParticle = Ball(space, (x,y), inRadius, inMass, inFriction)
                interiorParticles.append(interiorParticle)
            
    return bots, skinParticles, interiorParticles


def interiorPattern(systemRadius, inRadius, botRadius, percentInteriorRemove=0):
    R = systemRadius
    in_rings_radius = [] #**
    gran_per_ring = []
    
    buffer = .001 # Distance between rings 
    d_start = botRadius - inRadius # Distance between first ring and outer boxes
    if d_start<0: d_start = 0
    current_radius = R - (2*botRadius/2 + inRadius + d_start)
    current_circumference = 2*np.pi*current_radius
    
    while current_circumference > (2*inRadius)*3: # Not allowing less than 3 spheres in a ring
        in_rings_radius.append(current_radius)
        current_radius -= (2*inRadius + buffer)
        current_circumference = 2*np.pi*current_radius
        
    for radius in in_rings_radius:
        current_num_interior = floor((2*np.pi*radius)/(2*inRadius))
        gran_per_ring.append(current_num_interior)

    for num in in_rings_radius: 
        if num<0: 
            in_rings_radius.remove(num)
    for num in gran_per_ring: 
        if num<0: 
            gran_per_ring.remove(num)
    
    # Removing a percentage of the interior if too crowded
    numInterior = np.sum(gran_per_ring)
    numRemove = floor(numInterior*percentInteriorRemove)
    
    numRings= len(gran_per_ring)
    if numRemove==0: removePerRing=0
    else: removePerRing = numRemove//numRings
    removed = 0
    for ind, ring in enumerate(gran_per_ring):
        removedNow = ring - removePerRing
        if removedNow<0:
            removed += gran_per_ring[ind]
            gran_per_ring[ind]=0
        else:
            gran_per_ring[ind] -= removePerRing
            removed += removePerRing
        if removed>=numRemove: break
    
    return (gran_per_ring, in_rings_radius)

def createWalls(space, screenHeight, screenWidth, wallThickness, tunnel=False, env= None):
    # All values for this function are assumed to already be in Pixels
    wall_collision_type = 0
    
    # Bottom Wall
    body1 = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    shape1 = pymunk.Poly.create_box(body1, (screenWidth,wallThickness))
    shape1.body.position = (screenWidth//2,wallThickness//2)
    shape1.collision_type = wall_collision_type
    
    # Top Wall
    body2 = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    shape2 = pymunk.Poly.create_box(body2, (screenWidth,wallThickness))
    shape2.body.position = (screenWidth//2, screenHeight-wallThickness//2)
    shape2.collision_type = wall_collision_type
    
    # Back wall
    body3 = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    shape3 = pymunk.Poly.create_box(body3,(wallThickness, screenHeight))
    shape3.body.position = (wallThickness//2, screenHeight//2)
    shape3.collision_type = wall_collision_type

    # Front Wall
    body4 = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    shape4 = pymunk.Poly.create_box(body4,(wallThickness, screenHeight))
    shape4.body.position = (screenWidth, screenHeight//2)
    shape4.collision_type = wall_collision_type

    space.add(shape1,body1,shape2,body2,shape3,body3,shape4,body4)

    return [[shape1, body1], [shape2, body2], [shape3, body3], [shape4, body4]]


def pin_square(space, square):
    """
    Pins the square, wherever it is, to rotate about its center
    """
    center_pos = square.body.position
    pin = pymunk.Body(body_type=pymunk.Body.STATIC)
    pin.position = center_pos

    pin_constraint = pymunk.PinJoint(
        square.body, pin, (0,0), (0,0)
    )

    space.add(pin_constraint)

def bind_object(space, object_body, wall_body, length, spring_K):
    """
    Note: This function assumes the object is a SQUARE.
          Future iterations of the function should remove this assumption
    Inputs:
        - space: pymunk.Space which the objects exist in
        - object: pymunk.Body that must be bounded
        - wall: pymunk.Body which the object will be bound to
        - height: height of environment
        - width: width of environment
        - length: length of the object

    Returns:
        - None
    """
    spring_B = .5
    spring_RL = 0

    upper_spring = pymunk.DampedSpring(object_body,         # Body A
                                          wall_body,        # Body B
                                          (0, length/2),    # Anchor point rel to body A
                                          (0, length/2),    # Anchor point rel to body B
                                          spring_RL,        # Spring rest length
                                          spring_K,         # Spring stiffness
                                          spring_B          # Spring damping
                                        )

    lower_spring = pymunk.DampedSpring(object_body,         # Body A
                                          wall_body,        # Body B
                                          (0, -length/2),   # Anchor point rel to body A
                                          (0, -length/2),   # Anchor point rel to body B
                                          spring_RL,        # Spring rest length
                                          spring_K,         # Spring stiffness
                                          spring_B          # Spring damping
                                        )

    space.add(upper_spring, lower_spring)
    return [upper_spring, lower_spring]

def add_sliding_friction(space, system, mu):
    """
    Will iterate through system and add a simulated sliding friction
    """
    static_body = space.static_body
    g = 9.81 # Getting the acceleration due to gravity

    for obj in system:
        body = obj.body
        mass = body.mass
        friction_force = g*mass*mu

        # Create constraint for friction force
        pivot = pymunk.PivotJoint(static_body, body, (0,0), (0,0))
        space.add(pivot)
        pivot.max_bias = 0 # Disable joint correction
        pivot.max_force = friction_force

        # Create a constraint for sliding friction
        gear = pymunk.GearJoint(static_body, body, 0.0, 1.0)
        space.add(gear)
        gear.max_bias = 0 # Disable joint correctioon
        gear.max_force = friction_force


def createSlot(space, squareLength, friction, squarePosition, wallThickness, report_slot=False):
    """
    Will create a slot to get the square into. 
    The target is thus located at the center of this slot.
    """
    squareX, squareY = squarePosition
    buffer = squareLength/6  # A small buffer to get the square into
    slot_X = 2*squareLength 

    # Wall positions
    top_wall_pos = squareX + slot_X, squareY + squareLength/2 + wallThickness/2 + buffer/2
    bottom_wall_pos = squareX + slot_X, squareY - squareLength/2 - wallThickness/2 - buffer/2
    back_wall_pos = squareX + slot_X + squareLength/2, squareY 

    bodies = []
    shapes = []

    # Create walls
    top_body = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    top_shape = pymunk.Poly.create_box(top_body,(squareLength,wallThickness))
    top_shape.body.position = top_wall_pos
    bodies.append(top_body)
    shapes.append(top_shape)

    bottom_body = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    bottom_shape = pymunk.Poly.create_box(bottom_body,(squareLength,wallThickness))
    bottom_shape.body.position = bottom_wall_pos
    bodies.append(bottom_body)
    shapes.append(bottom_shape)

    back_body = pymunk.Body(0,0,body_type=pymunk.Body.STATIC)
    back_shape = pymunk.Poly.create_box(back_body,(wallThickness, squareLength+buffer))
    back_shape.body.position = back_wall_pos
    bodies.append(back_body)
    shapes.append(back_shape)

    # Add to space
    for body in bodies:
        space.add(body)
    for shape in shapes:
        shape.collision_type = 1
        shape.friction = friction
        shape.color = (0,0,0,150)
        space.add(shape)

    targetLoc = bottom_wall_pos[0], squareY

    if report_slot:
        top_wall_points = [
            [top_wall_pos[0] - squareLength/2, top_wall_pos[1]], # Point farthest left
            [top_wall_pos[0] + squareLength/2, top_wall_pos[1]]  # Point farthest right
        ]
        bottom_wall_points = [
            [bottom_wall_pos[0] - squareLength/2, bottom_wall_pos[1]], # Point farthest left
            [bottom_wall_pos[0] + squareLength/2, bottom_wall_pos[1]]  # Point farthest right
        ]
        back_wall_points = [
            [bottom_wall_pos[0] + squareLength/2, bottom_wall_pos[1]], # Bottom point
            [top_wall_pos[0] + squareLength/2, top_wall_pos[1]]        # Top point
        ]
        slot_points = [top_wall_points, bottom_wall_points, back_wall_points]
        return targetLoc, np.asarray(slot_points)
    else:
        return targetLoc

class Ball:
    def __init__(self, space, position, radius, mass, friction, collisionType = 0, color = (0,255,0,255)):
        self.body = pymunk.Body()
        self.radius = radius
        self.body.position = position
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.mass = mass
        self.shape.color = color
        self.shape.friction = friction
        self.shape.collision_type = collisionType
        space.add(self.body, self.shape)


"""
Potential field functions
"""
class controller:
    def __init__(self, pot_field_controller, noise=False):
        """
        Pot field controller
        """
        self.noise = noise
        if self.noise:
            self.gauss = np.random.normal # For adding noise to the actuation
            self.scale = 2 # Standard deviation from Gaussian distribution to sample noise from
        self.pot_field_controller = pot_field_controller

    def get_action(self, _obs=None, return_action=False):
        """
        Given the observation from the environment, return the recommended action to take per the pot field.
        We don't need the observation, thus it is set as _obs.

        return_action=True: Will return the forces on each bot as a single vector (i.e. [Fx1, Fy1, Fx2, Fy2,...,Fyn])
        return_action=False: Will return separate vectors for the forces in the X and Y direction (i.e. Fx, Fy)
        """
        Fx, Fy = self.pot_field_controller.run_controller()
        action = [] 
        for i in range(Fx.size):
            action.append(Fx[i])
            action.append(Fy[i])
        # action = self.correct_force(Fx, Fy) # Correct the action to fit into the proper space if required.
        # Adding noise to the action
        if self.noise:
            action = self.add_noise(action)
        
        if return_action:
            return action
        else:
            return Fx, Fy

    def add_noise(self, action):
        """
        Given the action vector, noise will be added
        """

        action = np.asarray(action) # Verify that action is an array
        action += self.gauss(size=action.size, scale=self.scale)
        action = np.clip(action,-1,1)
        action = action.tolist()

        return action

    def correct_force(self, Fx, Fy):
        """
        The pot field contoller returns two vectors defining the force in x and y direction for each bot.
        This must be adjusted to b in the discrete format that the environment is currently setup in.
        This can be removed if the agent is allowed to apply continuous actions on the space.

        Inputs:
            - Fx (ndarray): The forces in X-direction for each bot, as calculated by the pot field
            - Fy (ndarray): The force in Y-direction for each bot, as calculated by the pot field

        Returns:
            - action (ndarray): The recommended action to take, passed onto env.step() function
        """

        n = len(Fx) # Get the number of bots

        """
        Discrete Action Mapping:
            0: -1N
            1:  0N
            2:  1N
        """
        action = []
        thresh = 0.3 # Threshold for when a force should be applied versus not

        # Iterate through the forces for each bot
        for i in range(n):
            
            # Force in X-Direction
            force_x = Fx[i]
            
            if np.abs(force_x) < thresh: # Too small of a force, don't apply any forces
                action.append(1)
            elif force_x > 0:          # Move forward
                action.append(2)
            elif force_x < 0:          # Move backward
                action.append(0)
            
            # Force in Y-Direction
            force_y = Fy[i]

            if np.abs(force_y) < thresh:
                action.append(1)
            elif force_y > 0:
                action.append(2)
            elif force_y < 0:
                action.append(0)

        return np.asarray(action)


def create_pot_field(env):
    """
    Takes an instance of PyMunk environment and returns a pot field controller
    
    This function is used to get the pot controller given environment paramerters
    It is setup as a function because we need the environment to be created before we can create the pot field,
    and that happens in the calling file.
    """
    fieldLen = np.max([env.width, env.height])
    potFieldParams={
        'a':env.convert.Meters2Pixels(env.squareLength*.25),
        'b':env.convert.Meters2Pixels(env.squareLength*.25),
        'px':env.convert.Meters2Pixels(env.objectPosition[0]),
        'py':env.convert.Meters2Pixels(env.objectPosition[1]),
        'theta':0,
        'fieldLen':env.convert.Meters2Pixels(fieldLen),
        'res':env.convert.Meters2Pixels(0.1)}

    env.phi = analyticField(**potFieldParams)

    PWM = 255 # must be a number 0 < PWM < 255
    w = 1     # Frequency in hertz of PWM
    tn = (PWM/255)/w
    potControlParams={'environment':env,
                        'w':w,
                        'tn':tn,
                        'phi':env.phi,  # The field itself
                        'alpha':1,      # Gain of potfield. THIS IS THE FORCE IT WILL APPLY TO EACH BOT
                        'beta':0}       # Damping term of potfield

    pot_field = PotControl(**potControlParams)

    return pot_field


class PotControl:
    def __init__(self, environment, w, tn, phi, alpha, beta):

        #### Initialize some parameters for all cases of the controller
        self.env = environment      # The environment above (PyMunk)
        self.w=w                    # frequency for PWM
        self.tn=tn                  # time interval for PWM
        self.T=0                    # internal time for PWM
        
        #### Parameters needed from environment
        self.numBots = self.env.numBots  # Number of Bots
        self.tstep = self.env.dt         # Environment Timestep
                   
        ##### POT_FIELD_GRAB PARAMETERS        
        self.phi=phi          # potential field object
        self.fnx=self.phi.df2x # gradient in x direction  
        self.fny=self.phi.df2y # gradient in the y direction      
        self.alpha=alpha      # proportional gain 
        self.beta=beta        # damping term               
        

    # run controller
    def run_controller(self):
        self.get_position()            # get the position of the bots
        # self.get_velocity()            # get the velocity of the bots
        self.xbv = self.ybv = np.zeros(self.numBots)
        FX, FZ=self.grab_controller() # run the grab controller
        FX, FZ = self.apply_force(FX,FZ)        # Gather whether we will apply force to bots, based on PWM
        return FX, FZ
        

    #  Grabbing a ball controller 
    def grab_controller(self):
        ''' This controller is used for grabbing a fixed object'''
        FX=[]
        FY=[]
        px = self.env.square.body.position.x
        py = self.env.square.body.position.y
        for i in range(self.numBots):
            Fy=self.fny(self.xb[i], self.yb[i], px, py, 0 )
            Fx=self.fnx(self.xb[i], self.yb[i], px, py, 0 )
            mag=np.linalg.norm([Fx,Fy])
            FXX=Fx/mag
            FYY=Fy/mag
            fy=-self.alpha*FYY-self.beta*self.ybv[i]
            fx=-self.alpha*FXX-self.beta*self.xbv[i]
            FX.append(fx)
            FY.append(fy)
        return(np.asarray(FX),np.asarray(FY))
                   

    def apply_force(self, FX, FZ):  
        self.T=self.tstep+self.T
        self.T = round(self.T,4)
        if self.T>0 and self.T<=self.tn:
            FX=FX
            FZ=FZ
        else:
            pass # Currently passing to avoid setting force to 0
            # FX=np.zeros(len(FX))
            # FZ=np.zeros(len(FX))
            
        if self.T>(1/self.w):
            self.T=0

        return FX, FZ
        
    # get current position         
    def get_position(self):
        self.xb=[]        
        self.yb=[]
        for i in range(self.numBots):
            self.xb.append(self.env.bots[i].body.position.x)
            self.yb.append(self.env.bots[i].body.position.y)
        

    # get current velocity       
    def get_velocity(self):
        self.xbv=[]
        self.ybv=[]
        for i in range(self.numBots):
            self.xbv.append(self.env.bots[i].body.velocity.x)
            self.ybv.append(self.env.bots[i].body.velocity.y)


class analyticField:
    def __init__(self, a, b, px, py, theta = 0, res = 0, fieldLen=0):
        """
        a, b: Radii in two directions
        px, py: Center location of pot field
        theta: Rotation of field (about x-axis)
        res: Resolution of field
        fieldLen: The length of the field. i.e. the amount in each Cartesian direction that the field will be calculated for.
        """
        self.a = a
        self.b = b
        self.fieldLen = fieldLen
        self.px = px
        self.py = py
        self.theta = theta
        self.res = res
        
    def d_xy(self, x, y, px, py, a, b, phi=0):
        first  = (((-px+x)*sin(phi)+(-py+y)*cos(phi))**2)/(b**2)
        second = (((-px+x)*cos(phi)-(-py+y)*sin(phi))**2)/(a**2)
        tot = sqrt(first+second)
        return tot

    def d_dx(self, x, y, px, py, a, b, phi=0):
        denom1 = (((-px+x)*sin(phi)+(-py+y)*cos(phi))**2)/(b**2)
        denom2 =  (((-px+x)*cos(phi)-(-py+y)*sin(phi))**2)/(a**2)
        denom = sqrt(denom1 + denom2)
        
        num1 = ((-px+x)*sin(phi)+(-py+y)*cos(phi))*sin(phi)/(b**2)
        num2 = ((-px+x)*cos(phi)-(-py+y)*sin(phi))*cos(phi)/(a**2)
        num = num1+num2
        return (num/denom)
    
    def d_dy(self, x, y, px, py, a, b, phi=0):
        denom1 = (((-px+x)*sin(phi)+(-py+y)*cos(phi))**2)/(b**2)
        denom2 =  ((-px+x)*cos(phi)-(-py+y)*sin(phi))**2/(a**2)
        denom = sqrt(denom1 + denom2)
        
        num1 = ((-px+x)*sin(phi)+(-py+y)*cos(phi))*cos(phi)/(b**2)
        num2 = ((-px+x)*cos(phi)-(-py+y)*sin(phi))*sin(phi)/(a**2)
        num = num1-num2
        return (num/denom)
    
    def df2x(self, x, y, px, py, phi=0):
        kwargs = {'x':x, 'y':y, 'px':px, 'py':py, 'a':self.a, 'b':self.b, 'phi':phi}
        result = 4*self.d_xy(**kwargs)**3*(log(self.d_xy(**kwargs)))**2*self.d_dx(**kwargs) + 2*self.d_xy(**kwargs)**3*log(self.d_xy(**kwargs))*self.d_dx(**kwargs)
        return result
    
    def df2y(self, x, y, px, py, phi=0):
        kwargs = {'x':x, 'y':y, 'px':px, 'py':py, 'a':self.a, 'b':self.b, 'phi':phi}
        result = 4*self.d_xy(**kwargs)**3*(log(self.d_xy(**kwargs)))**2*self.d_dy(**kwargs) + 2*self.d_xy(**kwargs)**3*log(self.d_xy(**kwargs))*self.d_dy(**kwargs)
        return result
    
    def f_xy(self, x, y, px, py, a, b, phi=0):
        kwargs = {'x':x, 'y':y, 'px':px, 'py':py, 'a':a, 'b':b, 'phi':phi}
        result = self.d_xy(**kwargs)**2 * log(self.d_xy(**kwargs))
        return result
    
    def getField(self):
        # Calculating some parameters:
        self.xmin = self.px - self.fieldLen
        self.xmax = self.px + self.fieldLen
        self.ymin = self.py - self.fieldLen
        self.ymax = self.py + self.fieldLen
        xcount = int(round((self.xmax-self.xmin)/self.res))
        ycount = int(round((self.ymax-self.ymin)/self.res))
        xp = np.linspace(self.xmin,self.xmax,xcount)
        yp = np.linspace(self.ymin,self.ymax,ycount)
        
        xx, yy = np.meshgrid( xp, yp )
        kwargs={'x':xx, 'y':yy, 'px':self.px, 'py':self.py, 'a':self.a, 'b':self.b, 'theta':self.theta}
        zz = self.f_xy(**kwargs)
        fy, fx = np.gradient(zz**2)
        fx /= np.max(fx)
        fy /= np.max(fy)
        self.fnx = self.df2x
        self.fny = self.df2y
        return None


from tqdm import tqdm

def calc_JAMoEBA_Radius(skinRadius, skinRatio, botRadius, numBots):
    """
    Inputs:
        - skinRadius (float): The radius of skin particles on system
        - skinRatio (int): Ratio of number of skin particles per bot
        - botRadius (float): The radius of bot particles on system
        - numBots (int): Number of bots in the system

    Returns:
        - R (float): Radius of the system given parameters
    """
    startDistance = skinRadius # The start distance between bots
    arcLength = 2*botRadius+skinRatio*(2*skinRadius)+(skinRatio+1)*startDistance
    theta = 2*np.pi/numBots
    R = arcLength/theta #**
    return R

class Convert:
    def __init__(self, conversion_ratio=100):
        """
        Parameters
        ----------
        conversion_ratio : float
            The conversion ratio between pixels and meters in the form (pixels/meter).
        """
        self.ratio = conversion_ratio
        
    def Pixels2Meters(self, num_pixels):
        return (num_pixels*(1/self.ratio))
    
    def Meters2Pixels(self, meters):
        return (meters*self.ratio)
    
    def SpringK2Pixels(self, springK):
        return (springK*(1/self.ratio))
    
    def Pixels2SpringK(self, springKPixels):
        return (springKPixels*self.ratio)


def flatten(l):
    """
    Given a list that may contain arrays and scalars, will return an unwrapped list
    """
    for item in l:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def createVideo(saveLoc, imgLoc, videoName, imgShape):
    print('\nCreating video...')
    import glob # For creating videos
    import cv2 # For creating videos
    from shutil import rmtree

    out = cv2.VideoWriter(saveLoc+videoName+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 40, imgShape)
    for file in tqdm(glob.glob(imgLoc+'*.jpg')):
        img = cv2.imread(file)
        out.write(img)
    out.release
    
    rmtree(imgLoc)
    print('Video Creation Complete')


def save_runtime(saveloc,  file_name, runtime):
    from datetime import datetime, timedelta
    """
    Inputs:
        - saveloc (str): Path (NOT file name!) where the runtime will be stored
        - file_name (str): Name to save the information under
        - runtime (float): Runtime (in seconds) of some program
    Given a runtime (in seconds), 
    will save the amount of time it took to run some program at location 'saveloc'
    """
    sec = timedelta(seconds=runtime)
    d = datetime(1,1,1) + sec

    with open(saveloc + file_name +'.txt','w') as f:
        f.write('DAYS:HOURS:MIN:SEC\n')
        f.write("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))
    