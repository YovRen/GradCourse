import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable

from collections import namedtuple
import gym
import numpy as np
from scipy import optimize
from tools import *
class Policy_Network(nn.Module):
    def __init__(self, obs_space, act_space):
        super(Policy_Network, self).__init__()
        self.affine1 = nn.Linear(obs_space, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, act_space)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, act_space))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

class Value_Network(nn.Module):
    def __init__(self, obs_space):
        super(Value_Network, self).__init__()
        self.affine1 = nn.Linear(obs_space, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

Transition = namedtuple('Transition', ('state', 'action', 'mask',
                                       'reward', 'next_state'))
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

class Skylark_TRPO():
    def __init__(self, env, alpha = 0.1, gamma = 0.6, 
                    tau = 0.97, max_kl = 1e-2, l2reg = 1e-3, damping = 1e-1):
        self.obs_space = 80*80
        self.act_space = env.action_space.n
        self.policy = Policy_Network(self.obs_space, self.act_space)
        self.value = Value_Network(self.obs_space)
        self.env = env
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount rate
        self.tau = tau          # 
        self.max_kl = max_kl
        self.l2reg = l2reg
        self.damping = damping

        self.replay_buffer = Memory()
        self.buffer_size = 1000
        self.total_step = 0
        

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_mean, _, action_std = self.policy(Variable(state))
        action = torch.normal(action_mean, action_std)   
        return action

    def conjugate_gradients(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x


    def linesearch(self, model,
                f,
                x,
                fullstep,
                expected_improve_rate,
                max_backtracks=10,
                accept_ratio=.1):
        fval = f(True).data
        print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(model, xnew)
            newfval = f(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                print("fval after", newfval.item())
                return True, xnew
        return False, x

    def trpo_step(self, model, get_loss, get_kl, max_kl, damping):
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean() # 平均散度

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

        stepdir = self.conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(model)
        success, new_params = self.linesearch(model, get_loss, prev_params, fullstep,
                                        neggdotstepdir / lm[0])
        set_flat_params_to(model, new_params)
        return loss

    def learn(self, batch_size=128):
        batch = self.replay_buffer.sample()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        values = self.value(Variable(states))

        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i] # 计算了折扣累计回报
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i] # V - Q state value的偏差
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i] # 优势函数 A

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            '''
            构建替代回报函数 L_\pi(\hat{\pi})
            '''
            set_flat_params_to(self.value, torch.Tensor(flat_params))
            for param in self.value.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.value(Variable(states))

            value_loss = (values_ - targets).pow(2).mean() # (f(s)-r)^2

            # weight decay
            for param in  self.value.parameters():
                value_loss += param.pow(2).sum() * self.l2reg # 参数正则项
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.value).data.double().numpy())

        # 使用 scipy 的 l_bfgs_b 算法来优化无约束问题
        flat_params, _, opt_info = optimize.fmin_l_bfgs_b(func=get_value_loss, x0=get_flat_params_from(self.value).double().numpy(), maxiter=25)
        set_flat_params_to(self.value, torch.Tensor(flat_params))

        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds =  self.policy(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            '''
            计算策略网络的loss
            '''
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.policy(Variable(states))
            else:
                action_means, action_log_stds, action_stds = self.policy(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            # -A * e^{\hat{\pi}/\pi_{old}}
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = self.policy(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        self.trpo_step(self.policy, get_loss, get_kl, self.max_kl, self.damping)


    def train(self, num_episodes, batch_size = 128, num_steps = 100):
        for i in range(num_episodes):
            state = self.env.reset()

            steps, reward, sum_rew = 0, 0, 0
            done = False
            while not done and steps < num_steps:
                state = preprocess(state)
                action = self.choose_action(state)
                action = action.data[0].numpy()
                action_ = np.argmax(action)
                # Interaction with Env
                next_state, reward, done, info = self.env.step(action_) 
                next_state_ = preprocess(next_state)
                mask = 0 if done else 1
                self.replay_buffer.push(state, np.array([action]), mask, reward, next_state_)
                if len(self.replay_buffer) > self.buffer_size:
                    self.learn(batch_size)

                sum_rew += reward
                state = next_state
                steps += 1
                self.total_step += 1
            print('Episode: {} | Avg_reward: {} | Length: {}'.format(i, sum_rew/steps, steps))
        print("Training finished.")

def preprocess(I):
    '''
    根据具体gym环境的state输出格式，具体分析
    '''
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

if __name__ == "__main__":
    use_ray = True

    num_episodes = 1000
    env = gym.make("Pong-v0").env
    # env.render()

    if use_ray:
        import ray
        from ray import tune
        tune.run(
            'PPO', # ray 框架不包含 TRPO
            config={
                'env': "Pong-v0",
                'num_workers': 1,
                # 'env_config': {}
            }
        )
    else:
        trpo_agent = Skylark_TRPO(env)
        trpo_agent.train(num_episodes)