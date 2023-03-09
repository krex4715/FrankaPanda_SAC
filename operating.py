import gymnasium as gym
import panda_gym
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



env = gym.make("PandaReachJointsDense-v3", render_mode="human")
observation, info = env.reset()




check1s = []
check2s = []
check3s = []
for _ in range(100):
    check_1 = observation["achieved_goal"]
    check_2 = observation["desired_goal"]
    check_3 = observation["observation"]


    observation, reward, terminated, truncated, info = env.step([0,0,0,0,0,0,0])

    if terminated or truncated:
        observation, info = env.reset()

    
    
    check1s.append(check_1)
    check2s.append(check_2)
    check3s.append(check_3)

    print(check_1)
    time.sleep(0.02)

check1s = np.array(check1s)
check2s = np.array(check2s)
check3s = np.array(check3s)


print(check1s.shape)
print(check2s.shape)
print(check3s.shape)



# plt.figure(figsize=(10,10))
# plt.plot(check1s)
# plt.show()

# plt.figure(figsize=(10,10))
# plt.plot(check2s)
# plt.show()

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(check3s[:,0],'r')
plt.plot(check3s[:,1],'g')
plt.plot(check3s[:,2],'b')
plt.subplot(2,1,2)
plt.plot(check3s[:,3],'r')
plt.plot(check3s[:,4],'g')
plt.plot(check3s[:,5],'b')
plt.show()






env.close()