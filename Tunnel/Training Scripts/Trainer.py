# %% 
# Main cell

if __name__ == '__main__':
    # Importing Environment and environment dependencies
    from env_params import *

    # Importing RL parameters and dependencies
    from RL_params_tf1 import *

    # Creating directory to save all results in
    mainDirectory = str(pathlib.Path(__file__).parent.absolute()) # Get the path of this file
    savefile = mainDirectory + '\\Experiment {} {}\\'.format(str(experimentNum), date.today())
    os.makedirs(savefile, exist_ok=True)
    
    # Importing and saving these files
    import env_params, RL_params_tf1
    copyfile(master_env.__file__,savefile+'Environment.py'.format(str(experimentNum)))
    copyfile(RL_params_tf1.__file__,savefile+'RL_params_tf1.py'.format(str(experimentNum)))
    copyfile(env_params.__file__,savefile+'env_params.py'.format(str(experimentNum)))
    copyfile(__file__, savefile+'PPO_Trainer_tf1.py'.format(str(experimentNum)))
   
    # Setting up Callback
    checkpoint_callback = CheckpointCallback2(
        save_freq=check_freq, 
        save_path = savefile 
    ) 

    # Assert that we are not gathering simulation information during training
    envParams['dataCollect']=False
    envParams['saveVideo']=False

    # Creating multiple environments for multiple workers
    env = make_vec_env(pymunkEnv, n_envs= nEnvs, env_kwargs=envParams, vec_env_cls=SubprocVecEnv, monitor_dir = savefile)

    # # Create the model with expert trajectories
    # model = PPO2("MlpPolicy", env, verbose=1,
    #                 gamma=gamma, 
    #                 n_steps = n_steps, 
    #                 ent_coef = ent_coef,
    #                 learning_rate = learning_rate,
    #                 vf_coef = vf_coef,
    #                 nminibatches = nminibatches,
    #                 noptepochs = noptepochs,
    #                 cliprange = cliprange,
    #                 tensorboard_log = savefile,
    #                 seed = seed,
    #                 policy_kwargs=policy_kwargs)

    # Loading a model which was already started
    model = PPO2.load(
        'Experiment 16b 2022-01-10/rl_model_21650000_steps.zip',
        env=env,
        tensorboard_log = savefile
    )
    
    # Train the model
    learn_start = time.time()
    model.learn(total_timesteps=training_timesteps, callback=checkpoint_callback)
    learn_end = time.time()
    
    # Save information on how long training took
    runtime = learn_end - learn_start
    save_runtime(savefile, 'Training_Time', runtime)
    
    # Save the model
    model.save(savefile + experimentName + '_agent')

    # Close the environment
    env.close()

    # Delete the model and environment
    # This is done to verify that, if there are multiple trainings being done on a computer, 
    # that the correct agent and environment are paired.
    del model
    del env
    
    # Model Evaluation
    # We will now test the agent!
    if test:
        os.chdir(savefile)
    
        model = PPO2.load(experimentName+'_agent.zip')
        envParams['dataCollect']=True
        envParams['saveVideo']=True
        
        for i in range(num_tests):
            name = experimentName+'_v{}'.format(str(i+1))
            envParams['experimentName']=savefile + name
            Env=pymunkEnv(**envParams)
            obs=Env.reset()
            
            for j in range(time_per_test):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = Env.step(action)
            
                if render: 
                    Env.render()
                    
                if done: 
                    break
            
            if Env.dataCollect:
                Env.dataExport() # Export the data from the simulation
                plt.close('all')
            if Env.saveVideo:
                createVideo(savefile,Env.videoFolder,name, (width, height))
            Env.close()