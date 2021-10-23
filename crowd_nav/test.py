import gym
import argparse
import numpy as np
import configparser
import itertools
import logging
import os
import shutil
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.explorer import Explorer
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *

import matplotlib.pyplot as plt

def config_file_path():
    current_path = os.path.dirname(__file__)
    env_config_file = current_path + '/configs/env1.config'
    policy_config_file = current_path + '/configs/policy.config'
    return env_config_file,policy_config_file

def main():

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    # parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--test_stage', type=int,default=1)
    parser.add_argument('--visualize', default=False, action='store_true')
    args = parser.parse_args()

    if args.test_stage==1:
        args.env_config="configs/env.config"
        model_file="rl_model.pth"
    elif args.test_stage==2:
        args.env_config = "configs/env1.config"
        model_file="finetuned_model.pth"

    current_path = os.path.dirname(__file__)
    env_config_file=os.path.join(current_path,args.env_config)
    policy_config_file=os.path.join(current_path,args.policy_config)

    #log show
    logging.getLogger().setLevel(logging.INFO)

    #device config
    device=torch.device('cuda')

    #config file path
    #
    # env_config_file, policy_config_file=config_file_path()
    model_weights_path=os.path.join(args.output_dir,model_file)

    if os.path.exists(model_weights_path)==False:
        raise Exception("model not exist")


    #config policy
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file,encoding='utf-8')
    policy=policy_factory['ssdrl']()
    policy.configure(policy_config)
    policy.set_phase('test')
    policy.set_device(device)

    if policy.trainable:
        policy.get_model().load_state_dict(torch.load(model_weights_path))

    #config env
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file,encoding='utf-8')
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    #env.test_sim='circle_crossing'

    robot=Robot(env_config,'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer=Explorer(env,robot,device,gamma=0.9)



    policy.set_env(env)
    robot.print_info()
    reachgoalcase = 0
    collisioncase = 0
    timeoutcase = 0
    stress_list = []
    time_list=[]
    for i in range(env.case_size['test']):
        ob=env.reset('test',None)
        done=False
        last_pos=np.array(robot.get_position())
        rewards = []
        while not done:
            action=robot.act(ob)
            ob, reward, done, info = env.step(action)
            current_pos=np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos

            rewards.append(reward)


        cumulative_rewards=sum([pow(0.9, t * env.robot.time_step * env.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)])
        #print(cumulative_rewards)
        if isinstance(info, ReachGoal):
            reachgoalcase = reachgoalcase + 1
            stress_list.append(env.global_nervous[-1])
            time_list.append(env.global_time)
        if isinstance(info, Collision):
            collisioncase = collisioncase + 1
            stress_list.append(-1)
            time_list.append(-1)
        if isinstance(info, Timeout):
            timeoutcase = timeoutcase + 1
            stress_list.append(env.global_nervous[-1])
            time_list.append(env.global_nervou)

        print(info,stress_list[-1])

        if args.visualize:
            env.render(mode='video')

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) /  len(human_times))

        if i>5:break

    print("****************")

    print("success rate:",reachgoalcase/float(env.case_size['test']))

    stress_list=[i for i in stress_list  if i>=0]
    print("avg socail stress",sum(stress_list)/len(stress_list))

    time_list=[i for i in time_list  if i>=0]
    print("avg nav time",sum(time_list)/len(time_list))



if __name__=='__main__':
    main()