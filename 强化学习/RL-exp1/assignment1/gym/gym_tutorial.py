import gym
env = gym.make('Blackjack-v0')
for i_episode in range(20):
    observation = env.reset() #初始化环境每次迭代
    for t in range(100):
        env.render() #显示
        print(observation)
        action = env.action_space.sample() #随机选择action
        observation, reward, done,  _= env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()