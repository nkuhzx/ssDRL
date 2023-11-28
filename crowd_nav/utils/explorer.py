import logging
import copy
import torch
from crowd_sim.envs.utils.info import *


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy


    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

            if imitation_learning:
                print("\r \033[0;33mThe IL has processed {:.3%} Average reward {:.3f} Success {:.2f} Collision {:.2f} Timeout {:.2f}".format((i + 1)/k,average(cumulative_rewards),success/(i+1),collision/(i+1),timeout/(i+1)), end="")
            elif phase=='val':
                print("\r \033[0;33mThe RL evaluation has processed {:.3%} Average reward {:.3f} Success {:.2f} Collision {:.2f} Timeout {:.2f}".format((i + 1)/k,average(cumulative_rewards),success/(i+1),collision/(i+1),timeout/(i+1)), end="")

            #print(average(cumulative_rewards),success,collision,timeout)
        print("\n \033[0m")
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i + 1])
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i + 1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0

            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)
            # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     if state.is_cuda:
            #         padding=padding.cuda()
            #     state = torch.cat([state, padding])

            self.memory.push((state, value,reward,next_state))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
