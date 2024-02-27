<div align='center'>强化学习 Lab2 DQN</div>

# 实验要求
* 基于助教给出的代码，完善DQN算法的实现
    * 一共有3处TODO需要你补全
* 在此基础上，实现Double DQN (DDQN), Dueling DQN, Dueling DDQN，并对DQN和它们的表现进行比较
    * 绘制Reward曲线（4条：DQN, DDQN, Dueling DQN, Dueling DDQN）。为了更好的视觉效果，可以从Tensorboard中导出CSV，用seaborn等重绘
    * 进行简要的分析，包括收敛速度、最优性、稳定性等角度
    * 录制各方法最好策略的视频，10秒以内
        * ubuntu下可使用kazam
* 不需要太过关注训练的分数
* 加分项：
    * 实现Rainbow中其他改进手段，并进行对比
        * Prioritized Replay
        * Multi-Step
        * Noisy-Net
        * …
        * 可参考arXiv: 1710.02298 (https://arxiv.org/pdf/1710.02298.pdf)

