import sys
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
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.explorer import Explorer

from crowd_sim.envs.utils.robot import Robot


def main():

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--train_stage', type=int,default=1)
    args = parser.parse_args()


    # manual setting
    args.policy='ssdrl'
    args.env_config='configs/env1.config'
    args.gpu=True
    args.train_stage=2

    if args.train_stage==1:
        args.env_config = 'configs/env.config'
    elif args.train_stage==2:
        args.env_config = 'configs/env1.config'
    #args.resume=True

    # configure paths

    # reinforcement learning stage one
    # train based on env.config
    if args.train_stage==1:

        make_new_dir = True
        if os.path.exists(args.output_dir):
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
                args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
                args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
                args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))


        if make_new_dir:
            os.makedirs(args.output_dir)
            shutil.copy(args.env_config, args.output_dir)
            shutil.copy(args.policy_config, args.output_dir)
            shutil.copy(args.train_config, args.output_dir)
        log_file = os.path.join(args.output_dir, 'output.log')
        il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
        rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')


        # configure logging
        mode = 'a' if args.resume else 'w'
        file_handler = logging.FileHandler(log_file, mode=mode)
        stdout_handler = logging.StreamHandler(sys.stdout)
        level = logging.INFO if not args.debug else logging.DEBUG
        logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                            format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)

        # configure policy
        policy = policy_factory[args.policy]()
        if not policy.trainable:
            parser.error('Policy has to be trainable')
        if args.policy_config is None:
            parser.error('Policy config has to be specified for a trainable network')
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        policy.configure(policy_config)
        policy.set_device(device)

        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        env = gym.make('CrowdSim-v0')
        env.configure(env_config)
        robot = Robot(env_config, 'robot')
        logging.info('Robot sensor: %s ,sensor range: %d' %(robot.sensor,robot.sensing_range))
        env.set_robot(robot)

        # read training parameters
        if args.train_config is None:
            parser.error('Train config has to be specified for a trainable network')
        train_config = configparser.RawConfigParser()
        train_config.read(args.train_config)
        rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
        train_batches = train_config.getint('train', 'train_batches')
        train_episodes = train_config.getint('train', 'train_episodes')
        sample_episodes = train_config.getint('train', 'sample_episodes')
        target_update_interval = train_config.getint('train', 'target_update_interval')
        evaluation_interval = train_config.getint('train', 'evaluation_interval')
        capacity = train_config.getint('train', 'capacity')
        epsilon_start = train_config.getfloat('train', 'epsilon_start')
        epsilon_end = train_config.getfloat('train', 'epsilon_end')
        epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
        checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

        # configure trainer and explorer
        memory = ReplayMemory(capacity)
        model = policy.get_model()
        batch_size = train_config.getint('trainer', 'batch_size')
        trainer = Trainer(model, memory, device, batch_size)
        explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

        # imitation learning
        if args.resume:
            if not os.path.exists(rl_weight_file):
                logging.error('RL weights does not exist')
            model.load_state_dict(torch.load(rl_weight_file))
            rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
            logging.info('Load reinforcement learning trained weights. Resume training')
        elif os.path.exists(il_weight_file):
            model.load_state_dict(torch.load(il_weight_file))
            logging.info('Load imitation learning trained weights.')
        else:
            il_episodes = train_config.getint('imitation_learning', 'il_episodes')
            il_policy = train_config.get('imitation_learning', 'il_policy')
            il_epochs = train_config.getint('imitation_learning', 'il_epochs')
            il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
            trainer.set_learning_rate(il_learning_rate)
            if robot.visible:
                safety_space = 0
            else:
                safety_space = train_config.getfloat('imitation_learning', 'safety_space')
            il_policy = policy_factory[il_policy]()
            il_policy.multiagent_training = policy.multiagent_training
            il_policy.safety_space = safety_space
            robot.set_policy(il_policy)
            explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
            trainer.optimize_epoch(il_epochs)
            torch.save(model.state_dict(), il_weight_file)
            logging.info('Finish imitation learning. Weights saved.')
            logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
        explorer.update_target_model(model)

        # reinforcement learning stage one
        policy.set_env(env)
        robot.set_policy(policy)
        robot.print_info()
        trainer.set_learning_rate(rl_learning_rate)
        # fill the memory pool with some RL experience
        if args.resume:
            robot.policy.set_epsilon(epsilon_end)
            explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
            logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
        episode = 0

        while episode < train_episodes:
            if args.resume:
                epsilon = epsilon_end
            else:
                if episode < epsilon_decay:
                    epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
                else:
                    epsilon = epsilon_end
            robot.policy.set_epsilon(epsilon)

            # evaluate the model
            if episode % evaluation_interval == 0:
                explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

            # sample k episodes into memory and optimize over the generated memory
            explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
            trainer.optimize_batch(train_batches)
            episode += 1

            if episode % target_update_interval == 0:
                explorer.update_target_model(model)

            if episode != 0 and episode % checkpoint_interval == 0:
                torch.save(model.state_dict(), rl_weight_file)


    # reinforcement learning stage two
    # train based on env1.config according to rl_model.pth
    elif args.train_stage==2:

        make_new_dir = False
        # args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
        # args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
        # args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))


        log_file = os.path.join(args.output_dir, 'output.log')
        il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
        rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

        # configure logging
        mode = 'a'
        file_handler = logging.FileHandler(log_file, mode=mode)
        stdout_handler = logging.StreamHandler(sys.stdout)
        level = logging.INFO if not args.debug else logging.DEBUG
        logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                            format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)

        # configure policy
        policy = policy_factory[args.policy]()
        if not policy.trainable:
            parser.error('Policy has to be trainable')
        if args.policy_config is None:
            parser.error('Policy config has to be specified for a trainable network')
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        policy.configure(policy_config)
        policy.set_device(device)

        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        env = gym.make('CrowdSim-v0')
        env.configure(env_config)
        robot = Robot(env_config, 'robot')
        logging.info('Robot sensor: %s ,sensor range: %d' %(robot.sensor,robot.sensing_range))
        env.set_robot(robot)

        # read training parameters
        if args.train_config is None:
            parser.error('Train config has to be specified for a trainable network')
        train_config = configparser.RawConfigParser()
        train_config.read(args.train_config)
        rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
        train_batches = train_config.getint('train', 'train_batches')
        train_episodes = train_config.getint('train', 'train_episodes')
        sample_episodes = train_config.getint('train', 'sample_episodes')
        target_update_interval = train_config.getint('train', 'target_update_interval')
        evaluation_interval = train_config.getint('train', 'evaluation_interval')
        capacity = train_config.getint('train', 'capacity')
        epsilon_start = train_config.getfloat('train', 'epsilon_start')
        epsilon_end = train_config.getfloat('train', 'epsilon_end')
        epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
        checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

        # configure trainer and explorer
        memory = ReplayMemory(capacity)
        model = policy.get_model()
        batch_size = train_config.getint('trainer', 'batch_size')
        trainer = Trainer(model, memory, device, batch_size)
        explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file))
        logging.info('Load reinforcement learning trained weights. Resume training')

        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        il_weight_file = os.path.join(args.output_dir, 'finetuned_model.pth')
        torch.save(model.state_dict(), il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

        #update the target model
        explorer.update_target_model(model)


        policy.set_env(env)
        robot.set_policy(policy)
        robot.print_info()

        explorer.run_k_episodes(100, 'val', episode=0)

if __name__ == '__main__':
    main()