import gymnasium as gym
import panda_gym
import time
import buffer
import train
import numpy as np
import gc



env = gym.make('PandaReachJointsDense-v3', render_mode="rgb_array",renderer="OpenGL")
MAX_EPISODES = 1000
MAX_STEPS = 200
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = 6
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print('----------------')
print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)


ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
# threshold = 2850
# trainer.load_models(load_dir='Model_history_Joint/',episode=threshold)

avg_reward = 0
success_count=0


loss_actors = []
loss_critics = []
for _ep in range(MAX_EPISODES):
    # _ep = _ep+threshold
    observation, info = env.reset()

    print('EPISODE :- ', _ep)
    sum_reward=0
    
    for r in range(MAX_STEPS):
        state = np.float32(observation['desired_goal'].tolist() + observation["observation"][0:3].tolist())
        # state = observation['observation']
        if _ep%5 == 0:
            action = trainer.get_exploitation_action(state)
        else:
            action = trainer.get_exploration_action(state)

        new_observation, reward, terminated, truncated, info = env.step(action)
        
        
        if terminated:
            success_count+=1
            reward=5


        if terminated or truncated:
            new_state=None
        
        else:
            new_state = np.float32(new_observation['desired_goal'].tolist() + new_observation["observation"][0:3].tolist())
            # new_state = new_observation['observation']
            ram.add(state, action, reward, new_state)
        
        sum_reward +=reward
        

        
        observation = new_observation
        
        iteration, loss_actor, loss_critic = trainer.optimize()
        if iteration%100==0:
            loss_actors.append(loss_actor)
            loss_critics.append(loss_critic)
        
        # time.sleep(0.03)

        if terminated:
            break


            
    gc.collect()
    print("Average Reward :" , sum_reward)


    if _ep%50 == 0:
        trainer.save_models(save_dir='Model_history_Joint/' , episode_count=_ep)
        print(f'success " {success_count} / 50')

        success_count=0


print('Completed episodes')