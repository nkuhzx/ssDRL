import gym
import argparse
import numpy as np
import configparser
import itertools
import logging
import os
import shutil
import torch
import math

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.utils import AverageMeter
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from tqdm import tqdm

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

    # metric
    M_Success=AverageMeter()
    M_Perfect=AverageMeter()
    M_Comfort=AverageMeter()
    M_Time=AverageMeter()
    M_Score=AverageMeter()
    pbar=tqdm(total=env.case_size['test'])
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
            screen_output="ReachGoal"
        if isinstance(info, Collision):
            collisioncase = collisioncase + 1
            stress_list.append(-1)
            time_list.append(-1)
            screen_output = "Collision"
        if isinstance(info, Timeout):
            timeoutcase = timeoutcase + 1
            stress_list.append(env.global_nervous[-1])
            time_list.append(env.global_nervou)
            screen_output = "Timeout"

        # print(info,stress_list[-1])

        if args.visualize:
            env.render(mode='video')

        # logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        # if robot.visible and info == 'reach goal':
        #     human_times = env.get_human_times()
        #     logging.info('Average time for humans to reach goal: %.2f', sum(human_times) /  len(human_times))

        # if i>5:break

        # evaluation
        case_comfort=math.exp(stress_list[-1])
        case_performance=case_comfort*env.time_limit/time_list[-1]

        # update the Success rate
        if isinstance(info, ReachGoal):
            M_Success.update(1)
        else:
            M_Success.update(0)

        # update the Perfect rate
        if case_comfort==1:
            M_Perfect.update(1)
        else:
            M_Perfect.update(0)

        # update the avg comfort index
        M_Comfort.update(case_comfort)

        # update the avg time
        M_Time.update(time_list[-1])

        # update the avg performance score
        M_Score.update(case_performance)

        # print(M_Success.avg,M_Perfect.avg,M_Time.avg,M_Comfort.avg,M_Score.avg,info)
        pbar.set_description("Case: [{0}], info: {1}".format(i,screen_output))
        pbar.set_postfix(Success_rate=M_Success.avg,
                         Perfect_rate=M_Perfect.avg,
                         avg_Comfort=M_Perfect.avg,
                         avg_Time=M_Time.avg,
                         avg_Score=M_Score.avg)

        pbar.update(1)



if __name__=='__main__':
    main()