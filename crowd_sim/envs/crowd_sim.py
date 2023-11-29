import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import math
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.utils import Point,Vector,Line
from crowd_sim.envs.utils.utils import hr_intersection_area,collision_detected,hr_intersection_area_backup,hh_intersection_area
from crowd_sim.envs.policy.realscene import Realscene

#env
#policy

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.sim_type=None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.current_case_No=None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.realpolicy=None
        self.scene_width=None
        self.init_pos=None
        self.human_num = None
        self.dynamic_id_list=None
        # for visualization
        self.states = None
        self.eye_contact_states=None
        self.intention_states=None
        self.id_lists=None
        self.action_values = None
        self.attention_weights = None
        # for visual debug
        self.debugs=None

        #nervous calcaution
        self.hr_nervous=None
        self.nervous_spaces=None
        self.human_ns_paras=None
        self.global_nervous=None

        self.distance_reward=None
        self.rewardsum=None
        self.squezzebool=None

        self.behavior_attribute=None

    def configure(self, config):
        self.config = config
        self.sim_type=config.get('sim','sim_type')
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.behavior_attribute=config.getboolean('env','behavior_attribute')
        if self.config.get('humans', 'policy') == 'orca' and self.sim_type=='const':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')

        elif self.config.get('humans', 'policy') == 'orca' and self.sim_type=='varied':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')


        elif self.config.get('humans', 'policy')=='realscene':

            #robot init pos
            self.init_pos=(0,-6.2)

            #commoe realpolicy for human
            self.realpolicy=Realscene()
            self.realpolicy.configure(config)

            #set the size for train and test
            self.case_size = {'train': self.realpolicy.capacity, 'val': math.floor(0.1*self.realpolicy.capacity),
                              'test': self.realpolicy.capacity}

            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.scene_width=self.realpolicy.get_scene_width()

        else:
            raise NotImplementedError


        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        if self.realpolicy is not None:
            logging.info('In real scene with variable number of human.')
            logging.info('human number: changed with case counter')

        elif self.sim_type=='const':
            logging.info('In hand craft scene with const number of human.')
            logging.info('human number: {}'.format(self.human_num))

        elif self.sim_type =='varied':
            logging.info('In hand craft scene with variable number of human.')
            logging.info('maximum human number: {}'.format(self.human_num))

        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        if self.realpolicy is not None:
            logging.info('Scene width: {}'.format(self.scene_width))
        else:
            logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))
    def set_robot(self, robot):
        self.robot = robot

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        elif rule == 'dynamic_mixed':
            self.humans = []

            circle_human_num=np.random.randint(0,math.ceil(human_num/2))
            for i in range(circle_human_num):
                self.humans.append(self.generate_circle_crossing_human())
            square_human_num=human_num-circle_human_num
            for i in range(square_human_num):
                self.humans.append(self.generate_square_crossing_human())
        else:
            raise ValueError("Rule doesn't exist")

    def generate_real_scene_init_case_human_position(self,case_counter):
        self.humans=[]
        for i in range(len(self.dynamic_id_list)):
            self.humans.append(self.generate_real_scene_human(self.dynamic_id_list[i],case_counter,0))

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        human.hr_social_stress=0
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        human.hr_social_stress=0
        return human

    def generate_real_scene_human(self,id,case_counter,frame_step):

        human = Human(self.config, 'humans')
        collide = False
        while True:

            if collide==False :
                px, py = self.realpolicy.get_human_pos(id, case_counter, frame_step)
                original_px,original_py=px,py
            else:
                #print('original',original_px,original_py)
                px=px+(np.random.random()-0.5)/10.
                py=py+(np.random.random()-0.5)/10.
                #print('revised',px,py)

            collide=False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius:
                    collide = True

                    break
            if not collide:
                break

        human.set(px, py, 0, 0, 0, 0, 0)
        human.hr_social_stress=0
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

            # self.stat


        del sim
        return self.human_times


    def get_human_nervous_index(self,time):
        human_nervous_index = [0 for i in range(len(self.humans))]
        for i, human in enumerate(self.humans):

            human_nervous_index[i] = human.nervous_index(time)
        return human_nervous_index

    #caluate every ped's social stress
    def calculate_human_social_stress(self):
        for i, human in enumerate(self.humans):
            human.set_human_social_stress()

    #based every ped's social stress caluate the social that the robot expressed on ped
    def calculate_hr_social_stress(self,squeeze_table):
        for i,human in enumerate(self.humans):
            if human.squeeze_area != False:
                squeeze_index=list(np.where(squeeze_table[i]!=False)[0])
                squeeze_num=len(squeeze_index)
                ho_weight=[0 for num in range(squeeze_num)]
                for j in range(squeeze_num):
                    ho_weight[j]=human.pos.distance(self.humans[squeeze_index[j]].pos)
                ho_weight.append(human.pos.distance(self.robot.pos))
                ho_weight = np.array(ho_weight)
                ho_weight = 1 / ho_weight
                ho_weight=ho_weight/ho_weight.sum()
                hr_social_stress=human.stress_index*ho_weight[-1]
            else:
                hr_social_stress=human.stress_index

            human.set_hr_social_stress(hr_social_stress)

    #calculate the composite stress that the robot express
    def calculate_robot_composite_stress(self):
        hr_social_stress_list = [0 for i in range(len(self.humans))]
        hr_weight_list=[0 for i in range(len(self.humans))]
        for i, human in enumerate(self.humans):

            hr_social_stress_list[i] = human.hr_social_stress
            hr_weight_list[i]=human.pos.distance(self.robot.pos)

        hr_weight_list=np.array(hr_weight_list)
        hr_weight_list=1/hr_weight_list
        hr_weight_list=(hr_weight_list/hr_weight_list.sum())

        self.robot.set_robot_composite_stress((np.multiply(hr_weight_list,np.array(hr_social_stress_list))).sum())


    def get_all_hr_social_stress(self):
        hr_social_stress_list = [0 for i in range(len(self.humans))]
        for i, human in enumerate(self.humans):

            hr_social_stress_list[i] = human.hr_social_stress

        return hr_social_stress_list


    def get_robot_global_stress(self):
        return self.robot.hr_social_stress

    def stress_reward(self,k):
        reward=k*self.robot.hr_social_stress

        return reward


    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        # Initially judge whether the robot is set and the phase is right
        if self.robot is None:
            raise AttributeError('robot has to be set!')

        assert phase in ['train', 'val', 'test']

        # Configure the initial positions of pedestrians and robots
        # Real scene
        if self.realpolicy is not None:

            # set human number
            if self.case_counter[phase]>=0:
                self.human_num=len(self.realpolicy.get_human_id_list(self.case_counter[phase]))
            else:
                raise NotImplemented

            # set test case number if phase is 'test'
            if test_case is not None:
                self.case_counter[phase] = test_case

            # set global time and human time(not used in follow)
            self.global_time = 0
            if phase == 'test':
                self.human_times = [0] * self.human_num
            else:
                self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)

            # set the robot initial position
            self.robot.set(self.init_pos[0], self.init_pos[1],-self.init_pos[0], -self.init_pos[1],0, 0, np.pi / 2,radius=self.circle_radius)
            self.robot.visible=False

            if self.case_counter[phase]>=0:

                # set the human initial position
                self.dynamic_id_list=self.realpolicy.get_human_id_list(self.case_counter[phase],0)
                self.generate_real_scene_init_case_human_position(self.case_counter[phase])

                # set the current case number for step
                self.current_case_No=self.case_counter[phase]

                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

            else:
                raise NotImplemented

        # Scene with a fixed number of pedestrians
        elif self.sim_type=='const':

            if test_case is not None:
                self.case_counter[phase] = test_case
            self.global_time = 0
            if phase == 'test':
                self.human_times = [0] * self.human_num
            else:
                self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)

            if not self.robot.policy.multiagent_training:
                self.train_val_sim = 'circle_crossing'


            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        # Scene with a variable number of pedestrians
        elif self.sim_type=='varied':

            if test_case is not None:
                self.case_counter[phase] = test_case
            self.global_time = 0
            if phase == 'test':
                self.human_times = [0] * self.human_num
            else:
                self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)

            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)

                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

            else:
                raise NotImplemented

        elif self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError

        else:
            raise NotImplemented


        # config agent time step
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        # get the original nervous space of the agents
        self.robot.get_nervous_space()
        for human in self.humans:
            human.get_nervous_space()

        # get initial human number
        human_num=len(self.humans)

        # set the behavior attribute
        exist_waiting_id=np.random.choice(range(human_num))
        for i,human in enumerate(self.humans,0):
            human.sample_random_behavior_attributes(self.behavior_attribute,i==exist_waiting_id)


        # get initial squeeze situation (Matrix human_num*human_num)
        squeeze_table=np.zeros((human_num,human_num))
        for i in range(human_num):
            for j in range(i + 1, human_num):
                squeeze_index=hh_intersection_area(self.humans[i],self.humans[j],0.3)
                squeeze_table[i,j]=squeeze_index
                squeeze_table[j,i]=squeeze_index

        # Calculate the tension space after human-robot interaction
        for i,human in enumerate(self.humans):
            hr_intersection_area_backup(human, self.robot, 2)

        # Calculate three types of social stress
        self.calculate_human_social_stress()
        self.calculate_hr_social_stress(squeeze_table)
        self.calculate_robot_composite_stress()

        # Storage initialization for display
        self.hr_nervous=list()
        self.nervous_spaces=list()
        self.human_ns_paras=list()
        self.global_nervous=list()
        self.squezzebool=False
        self.states = list()
        self.eye_contact_states=list()
        self.intent_states=list()
        self.id_lists=list()
        self.rewardsum=0

        self.debugs=list()

        # Retain for future use
        self.distance_reward=0

        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]

        elif self.robot.sensor == 'PartialOb':

            ob = []
            for human in self.humans:
                dis = self.robot.pos.distance(human.pos)
                if dis <= self.robot.sensing_range:
                    ob.append(human.get_observable_state())

        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)


    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        Determine whether to collide
        """

        """
        Step for manual designed approach 
        """
        def static_env_step():
            nonlocal action

            human_actions = []
            for human in self.humans:
                # observation for humans is always coordinates
                ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
                if self.robot.visible:
                    ob += [self.robot.get_observable_state()]

                if self.behavior_attribute:
                    human.set_intention_state(self.global_time)
                    human.set_eye_contact_state(self.robot.get_observable_state())

                human_actions.append(human.act(ob))
                human.get_nervous_space()

            self.robot.get_nervous_space()

            human_num = len(self.humans)

            squeeze_table = np.zeros((human_num, human_num))
            squeeze_table_test = np.zeros((human_num, human_num + 1))


            for i in range(human_num):
                for j in range(i + 1, human_num):
                    squeeze_index = hh_intersection_area(self.humans[i], self.humans[j], 0.3)
                    squeeze_table[i, j] = squeeze_index
                    squeeze_table[j, i] = squeeze_index

                    squeeze_table_test[i, j] = squeeze_index
                    squeeze_table_test[j, i] = squeeze_index

                squeeze_table_test[i, -1] = self.humans[i].squeeze_area


            # collision detection
            dmin = float('inf')
            collision = False
            for i, human in enumerate(self.humans):
                px = human.px - self.robot.px
                py = human.py - self.robot.py
                if self.robot.kinematics == 'holonomic':
                    vx = human.vx - action.vx
                    vy = human.vy - action.vy

                else:
                    vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                    vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius

                collision_index = hr_intersection_area_backup(human, self.robot, 2)

                # collision_index=False
                if closest_dist < 0 or collision_index:
                    collision = True
                    logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist
                if human.squeeze_area == True:
                    self.squezzebool = True

            # collision detection between humans
            human_num = len(self.humans)
            for i in range(human_num):
                for j in range(i + 1, human_num):
                    dx = self.humans[i].px - self.humans[j].px
                    dy = self.humans[i].py - self.humans[j].py
                    dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                    if dist < 0:
                        # detect collision but don't take humans' collision into account
                        logging.debug('Collision happens between humans in step()')

            # check if reaching the goal
            end_position = np.array(self.robot.compute_position(action, self.time_step))
            reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

            last_pos = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position()))
            now_pos = norm(end_position - np.array(self.robot.get_goal_position()))
            self.distance_reward = last_pos - now_pos

            # the theta change in two step
            if self.robot.vx == 0 and self.robot.vy == 0:
                omega = 0
            else:
                next_theta = math.atan2(action.vy, action.vx)
                now_theta = math.atan2(self.robot.vy, self.robot.vx)
                omega = abs(next_theta - now_theta) / self.time_step

            # print(squeeze_table)
            self.calculate_human_social_stress()
            self.calculate_hr_social_stress(squeeze_table)
            self.calculate_robot_composite_stress()


            if self.global_time >= self.time_limit - 1:
                reward = 0
                done = True
                info = Timeout()
            elif collision:
                reward = self.collision_penalty
                done = True
                info = Collision()
            elif reaching_goal:

                reward = self.success_reward
                done = True
                info = ReachGoal()
            elif dmin < self.discomfort_dist:
                # only penalize agent for getting too close if it's visible
                # adjust the reward based on FPS
                #reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
                reward = self.stress_reward(-0.5)
                done = False
                info = Danger(dmin)
            else:
                # reward = 0
                reward = self.stress_reward(-0.5)
                done = False
                info = Nothing()

            if update:

                # print(self.nervous_reward_backup(1, self.time_step))
                if collision == False:
                    # store hr_nervous
                    # self.hr_nervous.append(self.get_human_nervous_index(self.time_step))
                    self.hr_nervous.append(self.get_all_hr_social_stress())
                    # print(self.get_human_nervous_index(self.time_step))
                    # store robot_nervous_space_para
                    self.nervous_spaces.append(self.robot.get_nervous_space())

                    # store human nervous_space_para
                    human_one_list = []
                    temp_num = 0
                    for human in self.humans:
                        human.set_nervous_space_para()

                        if temp_num == 0:
                            testchange = human.nervous_index(self.time_step)
                            # print(human.squeeze_area)

                        temp_human = np.row_stack((human.amid, human.bmid, human.aout, human.bout))
                        temp_human = temp_human.reshape((1, -1))[0]
                        human_one_list.append(temp_human)
                        temp_num = temp_num + 1
                        # print(human.amid,human.bmid,human.aout,human.bout)
                    human_one_list = np.array(human_one_list)
                    self.human_ns_paras.append(human_one_list)

                    self.rewardsum = self.rewardsum + self.get_robot_global_stress()
                    self.global_nervous.append(self.rewardsum)  # store state, action value and attention weights
                    self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
                    self.eye_contact_states.append([human.eye_contact for human in self.humans])
                    self.intent_states.append([human.intention for human in self.humans])
                    self.debugs.append([human.squeeze_area for human in self.humans])


                if hasattr(self.robot.policy, 'action_values'):
                    self.action_values.append(self.robot.policy.action_values)
                if hasattr(self.robot.policy, 'get_attention_weights'):
                    self.attention_weights.append(self.robot.policy.get_attention_weights())

                # update all agents
                self.robot.step(action)
                for i, human_action in enumerate(human_actions):
                    self.humans[i].step(human_action)
                self.global_time += self.time_step
                for i, human in enumerate(self.humans):
                    # only record the first time the human reaches the goal
                    if self.human_times[i] == 0 and human.reached_destination():
                        self.human_times[i] = self.global_time

                # compute the observation
                if self.robot.sensor == 'coordinates':
                    ob = [human.get_observable_state() for human in self.humans]

                elif self.robot.sensor=='PartialOb':

                    ob=[]
                    for human in self.humans:
                        dis=self.robot.pos.distance(human.pos)
                        if dis<=self.robot.sensing_range:
                            ob.append(human.get_observable_state())

                elif self.robot.sensor == 'RGB':
                    raise NotImplementedError
            else:
                self.get_all_hr_social_stress()
                if self.robot.sensor == 'coordinates':
                    ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]

                elif self.robot.sensor=='PartialOb':

                    if self.robot.kinematics == 'holonomic':
                        px = self.robot.px + action.vx*self.time_step

                        py = self.robot.py + action.vy*self.time_step

                    else:
                        px = self.robot.px + action.v * np.cos(action.r + self.robot.theta)*self.time_step
                        py = self.robot.py + action.v * np.sin(action.r + self.robot.theta)*self.time_step
                    robot_step_pos=Point(px,py)
                    ob=[]
                    for human, action in zip(self.humans, human_actions):
                        if human.kinematics == 'holonomic':
                            px = human.px + action.vx * self.time_step

                            py = human.py + action.vy * self.time_step

                        else:
                            px = human.px + action.v * np.cos(action.r + self.robot.theta) * self.time_step
                            py = human.py + action.v * np.sin(action.r + self.robot.theta) * self.time_step
                        human_step_pos=Point(px,py)
                        dis=robot_step_pos.distance(human_step_pos)-human.radius
                        if dis<=self.robot.sensing_range:
                            ob.append(human.get_next_observable_state(action))
                elif self.robot.sensor == 'RGB':
                    raise NotImplementedError

            return ob, reward, done, info

        def dynamic_env_step():
            pass

        def real_env_step():
            """
            Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

            Determine whether to collide
            """
            nonlocal action
            # get current frame step and id_list
            frame_step = round(self.global_time / self.time_step) + 1

            now_frame_id_list = self.realpolicy.get_human_id_list(self.current_case_No, frame_step)

            exit_id_list = list(set(self.dynamic_id_list) - set(now_frame_id_list))
            new_id_list = list(set(now_frame_id_list) - set(self.dynamic_id_list))
            subsist_id_list = list(set(now_frame_id_list).intersection(set(self.dynamic_id_list)))

            ## human robot collision detect
            # collsion with subsist human detect
            subsist_human_positions = []
            subsist_human_last_positions = []
            if frame_step > 0:
                subsist_human_last_positions = self.realpolicy.get_human_pos(subsist_id_list, self.current_case_No,
                                                                             frame_step - 1)
            else:
                subsist_human_last_positions = self.realpolicy.get_human_pos(subsist_id_list, self.current_case_No, 0)

            subsist_human_positions = self.realpolicy.get_human_pos(subsist_id_list, self.current_case_No, frame_step)

            dmin = float('inf')
            collision = False
            for i in range(len(subsist_id_list)):
                px = subsist_human_last_positions[i][0] - self.robot.px
                py = subsist_human_last_positions[i][1] - self.robot.py
                ex = subsist_human_positions[i][0] - (self.robot.px + self.time_step * action.vx)
                ey = subsist_human_positions[i][1] - (self.robot.py + self.time_step * action.vy)

                closest_dist = point_to_segment_dist(px, py, ex, ey, 0,
                                                     0) - self.realpolicy.human_radius - self.robot.radius
                if closest_dist < 0:
                    collision = True
                    logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist

            # collsion with new human detect
            new_human_position = self.realpolicy.get_human_pos(new_id_list, self.current_case_No, frame_step)
            for i in range(len(new_id_list)):
                now_pos = new_human_position[i]
                px = now_pos[0] - (self.robot.px + self.time_step * action.vx)
                py = now_pos[1] - (self.robot.py + self.time_step * action.vy)

                closest_dist = math.sqrt(math.pow(px, 2) + math.pow(py, 2)) - self.realpolicy.human_radius - self.robot.radius

                if closest_dist < 0:
                    collision = True
                    logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist

            # tension space update and determine if there is a collision
            self.robot.get_nervous_space()
            for human in self.humans:
                human.get_nervous_space()

            human_num = len(self.humans)

            squeeze_table = np.zeros((human_num, human_num))
            squeeze_table_test = np.zeros((human_num, human_num + 1))
            hh_intersection_area(self.humans[2], self.humans[1], 0.3)
            for i in range(human_num):
                for j in range(i + 1, human_num):
                    squeeze_index = hh_intersection_area(self.humans[i], self.humans[j], 0.3)
                    squeeze_table[i, j] = squeeze_index
                    squeeze_table[j, i] = squeeze_index

                    squeeze_table_test[i, j] = squeeze_index
                    squeeze_table_test[j, i] = squeeze_index

                squeeze_table_test[i, -1] = self.humans[i].squeeze_area

            collision_index=False
            for i, human in enumerate(self.humans):

                collision_index = hr_intersection_area_backup(human, self.robot, 2)
                if collision_index:
                    break


            collision = collision or collision_index

            # human collision detect
            now_frame_human_position = self.realpolicy.get_human_pos(now_frame_id_list, self.current_case_No,
                                                                     frame_step)
            human_num = len(now_frame_id_list)
            for i in range(human_num):
                now_pos1 = now_frame_human_position[i]

                for j in range(i + 1, human_num):
                    now_pos2 = now_frame_human_position[j]

                    dx = now_pos1[0] - now_pos2[0]
                    dy = now_pos1[1] - now_pos2[1]
                    dist = (dx ** 2 + dy ** 2) ** (1 / 2) - 2 * self.realpolicy.human_radius
                    if dist < 0:
                        # detect collision but don't take humans' collision into account
                        logging.debug('Collision happens between humans in step()')

            # check if reaching the goal
            end_position = np.array(self.robot.compute_position(action, self.time_step))
            reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

            self.calculate_human_social_stress()
            self.calculate_hr_social_stress(squeeze_table)
            self.calculate_robot_composite_stress()

            # calculate the reward
            if self.global_time >= self.time_limit - 1:
                reward = 0
                done = True
                info = Timeout()
            elif collision:
                reward = self.collision_penalty
                done = True
                info = Collision()
            elif reaching_goal:

                reward = self.success_reward
                done = True
                info = ReachGoal()
            elif dmin < self.discomfort_dist:
                # only penalize agent for getting too close if it's visible
                # adjust the reward based on FPS
                #reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
                reward = self.stress_reward(-0.5)
                done = False
                info = Danger(dmin)
            else:
                reward = self.stress_reward(-0.5)
                #reward = 0
                done = False
                info = Nothing()

            if update:

                if collision == False:

                    self.nervous_spaces.append(self.robot.get_nervous_space())

                    # store human nervous_space_para
                    human_one_list = []
                    temp_num = 0
                    for human in self.humans:
                        human.set_nervous_space_para()

                        if temp_num == 0:
                            testchange = human.nervous_index(self.time_step)
                            # print(human.squeeze_area)

                        temp_human = np.row_stack((human.amid, human.bmid, human.aout, human.bout))
                        temp_human = temp_human.reshape((1, -1))[0]
                        human_one_list.append(temp_human)
                        temp_num = temp_num + 1
                        # print(human.amid,human.bmid,human.aout,human.bout)
                    human_one_list = np.array(human_one_list)
                    self.human_ns_paras.append(human_one_list)

                    self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

                if hasattr(self.robot.policy, 'action_values'):
                    self.action_values.append(self.robot.policy.action_values)
                if hasattr(self.robot.policy, 'get_attention_weights'):
                    self.attention_weights.append(self.robot.policy.get_attention_weights())

                self.id_lists.append(self.dynamic_id_list.copy())
                # print(len(self.id_lists[-1]),len(self.states[-1][1]),len(self.human_ns_paras[-1]))

                self.rewardsum = self.rewardsum + self.get_robot_global_stress()
                self.global_nervous.append(self.rewardsum)  # store state, action value and attention weights
                # self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

                # self.debugs.append([human.squeeze_area for human in self.humans])

                # update all agent
                self.robot.step(action)

                # delet the exit human
                assert len(self.dynamic_id_list) == len(self.humans)
                for _, exit_id in enumerate(exit_id_list):
                    exit_id_index = self.dynamic_id_list.index(exit_id)
                    self.dynamic_id_list.remove(exit_id)
                    temp = self.humans.pop(exit_id_index)
                    del temp

                # add the new human
                for _, new_id in enumerate(new_id_list):
                    self.dynamic_id_list.append(new_id)
                    new_id_human = self.generate_real_scene_human(new_id, self.current_case_No, frame_step)
                    new_id_human.time_step = self.time_step
                    self.humans.append(new_id_human)

                # step the subsist human
                assert len(subsist_human_positions) == len(subsist_id_list)
                for subsist_id, subsist_human_pos in zip(subsist_id_list, subsist_human_positions):
                    subsist_id_index = self.dynamic_id_list.index(subsist_id)
                    self.humans[subsist_id_index].step(subsist_human_pos)
                self.global_time += self.time_step

                #print(self.current_case_No, frame_step)

                # record human reaching the goal first time

                if self.robot.sensor == 'coordinates':
                    ob = [human.get_observable_state() for human in self.humans]

                elif self.robot.sensor == 'PartialOb':

                    ob = []
                    for human in self.humans:
                        dis = self.robot.pos.distance(human.pos)-human.radius
                        if dis <= self.robot.sensing_range:
                            ob.append(human.get_observable_state())

                elif self.robot.sensor == 'RGB':
                    raise NotImplementedError

            else:
                # self.get_all_hr_social_stress()
                if self.robot.sensor == 'coordinates':
                    ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]

                elif self.robot.sensor=='PartialOb':

                    if self.robot.kinematics == 'holonomic':
                        px = self.robot.px + action.vx*self.time_step

                        py = self.robot.py + action.vy*self.time_step

                    else:
                        px = self.robot.px + action.v * np.cos(action.r + self.robot.theta)*self.time_step
                        py = self.robot.py + action.v * np.sin(action.r + self.robot.theta)*self.time_step
                    robot_step_pos=Point(px,py)

                    ob=[]
                    for id_num,pos in zip(now_frame_id_list,now_frame_human_position):
                        human_step_pos=Point(pos[0],pos[1])
                        dis = robot_step_pos.distance(human_step_pos) - self.realpolicy.human_radius
                        if dis<=self.robot.sensing_range:

                            # subsist pedestrian
                            if id_num in self.dynamic_id_list:
                                human=self.humans[self.dynamic_id_list.index(id_num)]
                                ob.append(human.get_next_observable_state(pos))
                            # new add pedestrian
                            else:
                                temp_human=Human(self.config, 'humans')
                                temp_human.set(pos[0],pos[1],0,0,0,0,0)
                                temp_human.hr_social_stress=0
                                ob.append(temp_human.get_observable_state())
                                del temp_human
                elif self.robot.sensor == 'RGB':
                    raise NotImplementedError

            return ob, reward, done, info

        if self.realpolicy is not None:
            return real_env_step()
        elif self.sim_type=='const':
            return static_env_step()
        elif self.sim_type=='varied':
            return static_env_step()

        else:
            raise NotImplemented


    def render(self, mode='human', output_file=None):

        from matplotlib import animation
        import matplotlib.pyplot as plt

        def static_env_render():
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

            x_offset = 0.11
            y_offset = 0.11
            cmap = plt.cm.get_cmap('hsv', 10)
            robot_color = 'yellow'
            goal_color = 'red'
            arrow_color = 'red'
            arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

            if mode == 'human':
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                for human in self.humans:
                    human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                    ax.add_artist(human_circle)
                ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
                plt.show()
            elif mode == 'traj':
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.tick_params(labelsize=16)
                ax.set_xlim(-5, 5)
                ax.set_ylim(-5, 5)
                ax.set_xlabel('x(m)', fontsize=16)
                ax.set_ylabel('y(m)', fontsize=16)

                robot_positions = [self.states[i][0].position for i in range(len(self.states))]
                human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                                   for i in range(len(self.states))]
                for k in range(len(self.states)):
                    if k % 4 == 0 or k == len(self.states) - 1:
                        robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                        humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                                  for i in range(len(self.humans))]
                        ax.add_artist(robot)
                        for human in humans:
                            ax.add_artist(human)
                    # add time annotation
                    global_time = k * self.time_step
                    if global_time % 4 == 0 or k == len(self.states) - 1:
                        agents = humans + [robot]
                        times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                          '{:.1f}'.format(global_time),
                                          color='black', fontsize=14) for i in range(self.human_num + 1)]
                        for time in times:
                            ax.add_artist(time)
                    if k != 0:
                        nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                                   (self.states[k - 1][0].py, self.states[k][0].py),
                                                   color=robot_color, ls='solid')
                        human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                       (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                       color=cmap(i), ls='solid')
                                            for i in range(self.human_num)]
                        ax.add_artist(nav_direction)
                        for human_direction in human_directions:
                            ax.add_artist(human_direction)
                plt.legend([robot], ['Robot'], fontsize=16)
                plt.show()

            elif mode == 'video':
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.tick_params(labelsize=16)
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                ax.set_xlabel('x(m)', fontsize=16)
                ax.set_ylabel('y(m)', fontsize=16)


                # add robot and its goal
                robot_positions = [state[0].position for state in self.states]

                vel_vx=[state[0].vx for state in self.states]
                vel_vy=[state[0].vy for state in self.states]
                robot_theta=np.degrees(np.arctan2(vel_vy,vel_vx))

                human_theta=[]
                for i in range(self.human_num+1):
                    if i==0:
                        pass
                    else:

                        vel_vx = [state[1][i-1].vx for state in self.states]
                        vel_vy = [state[1][i-1].vy for state in self.states]
                        human_theta.append(np.degrees(np.arctan2(vel_vy, vel_vx)))



                goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=10,
                                     label='Goal')
                robot_nervous_para=self.nervous_spaces


                robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)


                robot_nervous_space=patches.Arc(xy=robot_positions[0],width=2*robot_nervous_para[0][0],height=2*robot_nervous_para[0][1],angle=robot_theta[0],theta1=-90,theta2=90,color='red')



                ax.add_artist(robot_nervous_space)
                ax.add_artist(robot)
                ax.add_artist(goal)

                if self.robot.sensing_range is not None:
                    robot_sensor = plt.Circle(robot_positions[0], self.robot.sensing_range, fill=False, color='r',linestyle='-.')

                    ax.add_artist(robot_sensor)

                #ax.add_artist(robot_nervous_space)
                plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                          for i in range(len(self.humans))]


                #human nervous init
                humans_ns_midfl=[patches.Arc(xy=human_positions[0][i],width=2*self.human_ns_paras[0][i][0],height=2*self.human_ns_paras[0][i][2],angle=human_theta[i][0],theta1=0,theta2=90,color='black') for i in range(len(self.humans))]
                humans_ns_midfr=[patches.Arc(xy=human_positions[0][i],width=2*self.human_ns_paras[0][i][0],height=2*self.human_ns_paras[0][i][3],angle=human_theta[i][0],theta1=-90,theta2=0,color='black') for i in range(len(self.humans))]

                humans_ns_midbl = [patches.Arc(xy=human_positions[0][i], width=2 * self.human_ns_paras[0][i][1],
                                               height=2 * self.human_ns_paras[0][i][2], angle=human_theta[i][0], theta1=90,
                                               theta2=180, color='black') for i in range(len(self.humans))]

                humans_ns_midbr = [patches.Arc(xy=human_positions[0][i], width=2 * self.human_ns_paras[0][i][1],
                                              height=2 * self.human_ns_paras[0][i][3], angle=human_theta[i][0], theta1=180, theta2=-90,
                                              color='black') for i in range(len(self.humans))]

                humans_ns_outfl=[patches.Arc(xy=human_positions[0][i],width=2*self.human_ns_paras[0][i][4],height=2*self.human_ns_paras[0][i][6],angle=human_theta[i][0],theta1=0,theta2=90,color='blue') for i in range(len(self.humans))]
                humans_ns_outfr=[patches.Arc(xy=human_positions[0][i],width=2*self.human_ns_paras[0][i][4],height=2*self.human_ns_paras[0][i][7],angle=human_theta[i][0],theta1=-90,theta2=0,color='blue') for i in range(len(self.humans))]

                humans_ns_outbl = [patches.Arc(xy=human_positions[0][i], width=2 * self.human_ns_paras[0][i][5],
                                               height=2 * self.human_ns_paras[0][i][6], angle=human_theta[i][0], theta1=90,
                                               theta2=180, color='blue') for i in range(len(self.humans))]

                humans_ns_outbr = [patches.Arc(xy=human_positions[0][i], width=2 * self.human_ns_paras[0][i][5],
                                              height=2 * self.human_ns_paras[0][i][7], angle=human_theta[i][0], theta1=180, theta2=-90,
                                              color='blue') for i in range(len(self.humans))]

                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                          color='black', fontsize=12) for i in range(len(self.humans))]
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    ax.add_artist(human_numbers[i])

                    #ax.add_artist(humans_ns_midfl[i])
                    #ax.add_artist(humans_ns_midfr[i])
                    #ax.add_artist(humans_ns_midbl[i])
                    #ax.add_artist(humans_ns_midbr[i])

                    ax.add_artist(humans_ns_outfl[i])
                    ax.add_artist(humans_ns_outfr[i])
                    ax.add_artist(humans_ns_outbl[i])
                    ax.add_artist(humans_ns_outbr[i])

                # add time annotation
                time = plt.text(-1, 5.5, 'Time: {}'.format(0), fontsize=14)
                ax.add_artist(time)

                self.attention_weights=None
                self.hr_nervous=None
                # compute attention scores
                if self.attention_weights is not None:
                    attention_scores = [
                        plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                                 fontsize=16) for i in range(len(self.humans))]

                if self.hr_nervous is not None:
                    nervous_scores = [
                        plt.text(-5.5, -3.5 - 0.5 * i, 'Human {}: {:+.2f}'.format(i + 1, float(self.hr_nervous[0][i])),
                                 fontsize=14) for i in range(len(self.humans))]

                if self.global_nervous is not None:
                    global_scores=plt.text(-1, -5.5 , 'Social stress : {:.2f}'.format(self.global_nervous[0]*10),fontsize=14)

                # compute hr distance
                #hr_distance_show= [
                #    plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, ' {:.2f}'.format(float(self.hr_nervous[0][i])),
                #                 fontsize=16) for i in range(len(self.humans))]

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                if self.robot.kinematics == 'unicycle':
                    orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                                 state[0].py + radius * np.sin(state[0].theta))) for state
                                   in self.states]
                    orientations = [orientation]
                else:
                    orientations = []
                    for i in range(self.human_num + 1):
                        orientation = []
                        for state in self.states:
                            if i == 0:
                                agent_state = state[0]
                            else:
                                agent_state = state[1][i - 1]
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                 agent_state.py + radius * np.sin(theta))))
                        orientations.append(orientation)
                arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                          for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0

                def human_ns_update(humans_ns_,num_,human_position_,human_ns_paras_,human_theta_,frame_num_):
                    if humans_ns_==humans_ns_midfl:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][0]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][2]
                    elif humans_ns_==humans_ns_midfr:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][0]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][3]
                    elif humans_ns_==humans_ns_midbl:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][1]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][2]
                    elif humans_ns_==humans_ns_midbr:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][1]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][3]
                    elif humans_ns_==humans_ns_outfl:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][4]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][6]
                    elif humans_ns_==humans_ns_outfr:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][4]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][7]
                    elif humans_ns_==humans_ns_outbl:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][5]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][6]
                    elif humans_ns_==humans_ns_outbr:
                        humans_ns_[num_].width = 2 * human_ns_paras_[frame_num_][num_][5]
                        humans_ns_[num_].height = 2 * human_ns_paras_[frame_num_][num_][7]
                    else:
                        pass

                    humans_ns_[num_].center=human_position_[frame_num_][num_]
                    humans_ns_[num_].angle=human_theta_[num_][frame_num_]
                    #return(width_,height_,center_,angle_)

                def update(frame_num):
                    nonlocal global_step
                    nonlocal arrows
                    global_step = frame_num
                    robot.center = robot_positions[frame_num]
                    robot_nervous_space.center=robot_positions[frame_num]
                    if self.robot.sensing_range is not None:
                        robot_sensor.center=robot_positions[frame_num]

                    try:
                        robot_nervous_space.width=2*robot_nervous_para[frame_num][0]
                    except:
                        print(robot_nervous_para)
                        print(len(robot_nervous_para))
                        print(frame_num)
                    robot_nervous_space.height=2*robot_nervous_para[frame_num][1]
                    robot_nervous_space.angle=robot_theta[frame_num]

                    #update human_nervous_space
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                        if (self.eye_contact_states[frame_num][i]):
                            human_numbers[i].set_color('green')
                            human.set_color('green')
                        else:
                            human_numbers[i].set_color('black')
                            human.set_color('black')

                        if (self.intent_states[frame_num][i]==1):
                            human.set_alpha(0.3)
                            human.set_fill(True)
                        else:
                            human.set_alpha(None)
                            human.set_fill(False)

                        human_ns_update(humans_ns_midfl,i,human_positions,self.human_ns_paras,human_theta,frame_num)
                        human_ns_update(humans_ns_midfr, i, human_positions, self.human_ns_paras, human_theta,
                                             frame_num)
                        human_ns_update(humans_ns_midbl, i, human_positions, self.human_ns_paras, human_theta,
                                             frame_num)
                        human_ns_update(humans_ns_midbr, i, human_positions, self.human_ns_paras, human_theta,
                                             frame_num)
                        human_ns_update(humans_ns_outfl,i,human_positions,self.human_ns_paras,human_theta,frame_num)
                        human_ns_update(humans_ns_outfr, i, human_positions, self.human_ns_paras, human_theta,
                                             frame_num)
                        human_ns_update(humans_ns_outbl, i, human_positions, self.human_ns_paras, human_theta,
                                             frame_num)
                        human_ns_update(humans_ns_outbr, i, human_positions, self.human_ns_paras, human_theta,
                                             frame_num)

                        #update hr_distance
                        #hr_distance_show[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                        #hr_distance_show[i].set_text(' {:.2f}'.format(float(self.hr_nervous[frame_num][i])))


                        for arrow in arrows:
                            arrow.remove()
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                          arrowstyle=arrow_style) for orientation in orientations]
                        for arrow in arrows:
                            ax.add_artist(arrow)

                        if self.attention_weights is not None:
                            human.set_color(str(self.attention_weights[frame_num][i]))
                            attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                        if self.hr_nervous is not None:
                            human.set_color(str(abs(self.hr_nervous[frame_num][i])))
                            nervous_scores[i].set_text('human {}: {:+.2f}'.format(i, self.hr_nervous[frame_num][i]*10))

                    time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                    if self.global_nervous is not None:
                        if self.global_nervous[frame_num]>0:
                            global_scores.set_color('red')
                        else:
                            global_scores.set_color('blue')
                        global_scores.set_text('Social stress : {:+.2f}'.format(self.global_nervous[frame_num]*10))


                def plot_value_heatmap():
                    assert self.robot.kinematics == 'holonomic'
                    for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                        print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                                 agent.vx, agent.vy, agent.theta))

                    print(global_step)
                    print(self.debugs[global_step])



                    fig, axis = plt.subplots()
                    speeds = [0] + self.robot.policy.speeds
                    rotations = self.robot.policy.rotations + [np.pi * 2]
                    print(speeds,rotations)
                    r, th = np.meshgrid(speeds, rotations)
                    z = np.array(self.action_values[global_step % len(self.states)][1:])
                    z = (z - np.min(z)) / (np.max(z) - np.min(z))
                    z = np.reshape(z, (16, 5))
                    polar = plt.subplot(projection="polar")
                    polar.tick_params(labelsize=16)
                    mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                    plt.plot(rotations, r, color='k', ls='none')
                    plt.grid()
                    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                    cbar = plt.colorbar(mesh, cax=cbaxes)
                    cbar.ax.tick_params(labelsize=16)
                    plt.show()

                def on_click(event):
                    anim.running ^= True
                    if anim.running:
                        anim.event_source.stop()
                        if hasattr(self.robot.policy, 'action_values'):
                            plot_value_heatmap()
                    else:
                        anim.event_source.start()

                fig.canvas.mpl_connect('key_press_event', on_click)
                anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
                anim.running = True

                if output_file is not None:
                    #ffmpeg_writer = animation.writers['ffmpeg']
                    #writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                    #anim.save(output_file, writer=writer)
                    anim.save('test.gif', writer='pillow')
                else:
                    plt.show()
            else:
                raise NotImplementedError

        def dynamic_env_render():
            pass

        def real_env_render():
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

            x_offset = 0.11
            y_offset = 0.11
            cmap = plt.cm.get_cmap('hsv', 10)
            robot_color = 'gold'
            goal_color = 'red'
            arrow_color = 'red'
            arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
            ped_radius = self.realpolicy.human_radius

            if mode == 'human':
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_xlim(-7.5, 7.5)
                ax.set_ylim(-7.5, 7.5)
                for human in self.humans:
                    human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                    ax.add_artist(human_circle)
                ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
                plt.show()

            elif mode == 'traj':
                raise NotImplemented

            elif mode == 'video':
                fig, ax = plt.subplots(figsize=(14, 14))
                ax.tick_params(labelsize=16)
                lim_show = self.realpolicy.get_scene_width()

                ax.set_xlim(-lim_show, lim_show)
                ax.set_ylim(-lim_show, lim_show)
                ax.set_xlabel('x(m)', fontsize=16)
                ax.set_ylabel('y(m)', fontsize=16)

                # add robot and its goal
                robot_positions = [state[0].position for state in self.states]
                robot_nervous_para = self.nervous_spaces

                vel_vx = [state[0].vx for state in self.states]
                vel_vy = [state[0].vy for state in self.states]
                robot_theta = np.degrees(np.arctan2(vel_vy, vel_vx))

                # add robot and its goal
                goal = mlines.Line2D([0], [6], color=goal_color, marker='*', linestyle='None', markersize=10,
                                     label='Goal')
                robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
                robot_nervous_space = patches.Arc(xy=robot_positions[0], width=2 * robot_nervous_para[0][0],
                                                  height=2 * robot_nervous_para[0][1], angle=robot_theta[0], theta1=-90,
                                                  theta2=90, color='red')

                ax.add_artist(robot)
                ax.add_artist(goal)
                ax.add_artist(robot_nervous_space)

                if self.robot.sensing_range is not None:
                    robot_sensor = plt.Circle(robot_positions[0], self.robot.sensing_range, fill=False, color='r',linestyle='-.')

                    ax.add_artist(robot_sensor)

                # add humans and their id
                dynamic_id_dict = {}
                # get the init id list
                init_id_list = self.id_lists[0]

                # calcuate the human theta and pos
                human_positions = []
                human_theta = []
                human_ns_paras = self.human_ns_paras
                for state in self.states:
                    human_positions.append([state[1][j].position for j in range(len(state[1]))])

                    human_vel_vx = [state[1][i].vx for i in range(len(state[1]))]
                    human_vel_vy = [state[1][i].vy for i in range(len(state[1]))]
                    human_theta.append(np.degrees(np.arctan2(human_vel_vy, human_vel_vx)))

                for ped_id, ped_pos, ped_ns_para, ped_theta in zip(init_id_list, human_positions[0],
                                                                   self.human_ns_paras[0], human_theta[0]):
                    ped_circle_temp = plt.Circle(ped_pos, ped_radius, fill=False)
                    ped_num_temp = plt.text(ped_circle_temp.center[0] - x_offset, ped_circle_temp.center[1] - y_offset,
                                            str(ped_id), color='black', fontsize=8)

                    ped_ns_outfl_temp = patches.Arc(xy=(ped_pos[0], ped_pos[1]), width=2 * ped_ns_para[4],
                                                    height=2 * ped_ns_para[6], angle=ped_theta, theta1=0, theta2=90,
                                                    color='blue')
                    ped_ns_outfr_temp = patches.Arc(xy=(ped_pos[0], ped_pos[1]), width=2 * ped_ns_para[4],
                                                    height=2 * ped_ns_para[7], angle=ped_theta, theta1=-90, theta2=0,
                                                    color='blue')
                    ped_ns_outbl_temp = patches.Arc(xy=(ped_pos[0], ped_pos[1]), width=2 * ped_ns_para[5],
                                                    height=2 * ped_ns_para[6], angle=ped_theta, theta1=90, theta2=180,
                                                    color='blue')
                    ped_ns_outbr_temp = patches.Arc(xy=(ped_pos[0], ped_pos[1]), width=2 * ped_ns_para[5],
                                                    height=2 * ped_ns_para[7], angle=ped_theta, theta1=180, theta2=-90,
                                                    color='blue')

                    ax.add_artist(ped_circle_temp)
                    ax.add_artist(ped_num_temp)

                    ax.add_artist(ped_ns_outfl_temp)
                    ax.add_artist(ped_ns_outfr_temp)
                    ax.add_artist(ped_ns_outbl_temp)
                    ax.add_artist(ped_ns_outbr_temp)

                    dynamic_id_dict[ped_id] = [ped_circle_temp, ped_num_temp, ped_ns_outfl_temp, ped_ns_outfr_temp,
                                               ped_ns_outbl_temp, ped_ns_outbr_temp]

                # add time annotation
                time = plt.text(-1, 6.8, 'Time: {}'.format(0), fontsize=10)
                ax.add_artist(time)

                plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=10)

                # add arrow to show the dirction
                radius = self.robot.radius
                if self.robot.kinematics == 'unicycle':
                    orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                                 state[0].py + radius * np.sin(state[0].theta))) for
                                   state
                                   in self.states]
                    orientations = [orientation]
                else:
                    orientations = []

                    orientation = []
                    for state in self.states:
                        agent_state = state[0]

                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                               agent_state.py + radius * np.sin(
                                                                                   theta))))
                    orientations = [orientation]

                arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style) for
                          orientation in orientations]

                for arrow in arrows:
                    ax.add_artist(arrow)

                global_step = 0

                def update(frame_num):
                    nonlocal global_step
                    nonlocal dynamic_id_dict
                    nonlocal arrows

                    global_step = frame_num

                    # update robot pos and robot_nervous_space
                    robot.center = robot_positions[frame_num]
                    robot_nervous_space.center = robot_positions[frame_num]

                    robot_nervous_space.width = 2 * robot_nervous_para[frame_num][0]
                    robot_nervous_space.height = 2 * robot_nervous_para[frame_num][1]
                    robot_nervous_space.angle = robot_theta[frame_num]

                    if self.robot.sensing_range is not None:
                        robot_sensor.center=robot_positions[frame_num]

                    ## update human pos and id
                    now_id_list = self.id_lists[frame_num]
                    now_human_position = human_positions[frame_num]

                    now_human_theta = human_theta[frame_num]
                    now_human_ns_para = human_ns_paras[frame_num]

                    # now_id_list[frame_num]~human_positions[frame_num]

                    # remove the exit id from canvas
                    id_list_remove = list(set(dynamic_id_dict.keys()) - set(now_id_list))
                    for i in range(len(id_list_remove)):
                        remove_ped = dynamic_id_dict.get(id_list_remove[i])
                        remove_ped[0].remove()
                        remove_ped[1].set_text(None)
                        # del remove_ped[1]

                        remove_ped[1].remove()
                        for index in range(2, 6):
                            remove_ped[index].remove()

                        dynamic_id_dict.pop(id_list_remove[i])

                    # update the subsist pos
                    id_list_update = list(set(dynamic_id_dict.keys()) & set(now_id_list))
                    for _, id_update in enumerate(id_list_update):
                        update_ped = dynamic_id_dict.get(id_update)

                        pos_update = now_human_position[now_id_list.index(id_update)]
                        theta_update = now_human_theta[now_id_list.index(id_update)]
                        ns_para_updata = now_human_ns_para[now_id_list.index(id_update)]

                        update_ped[0].center = pos_update
                        update_ped[1].set_position((pos_update[0] - x_offset, pos_update[1] - y_offset))

                        for index in range(2, 6):

                            update_ped[index].center = pos_update
                            update_ped[index].angle = theta_update

                            if index < 4:
                                update_ped[index].width = 2 * ns_para_updata[4]
                            else:
                                update_ped[index].width = 2 * ns_para_updata[5]
                            if index % 2 == 0:
                                update_ped[index].height = 2 * ns_para_updata[6]
                            else:
                                update_ped[index].height = 2 * ns_para_updata[7]

                    # add the new id to canvas
                    id_list_add = list(set(now_id_list) - set(dynamic_id_dict.keys()))
                    for _, id_add in enumerate(id_list_add):
                        pos_add = now_human_position[now_id_list.index(id_add)]
                        theta_add = now_human_theta[now_id_list.index(id_add)]
                        ns_para_add = now_human_ns_para[now_id_list.index(id_add)]

                        ped_circle_temp = plt.Circle(pos_add, ped_radius, fill=False)
                        ped_num_temp = plt.text(ped_circle_temp.center[0] - x_offset,
                                                ped_circle_temp.center[1] - y_offset, str(id_add), color='black',
                                                fontsize=8)

                        ped_ns_outfl_temp = patches.Arc(xy=(pos_add[0], pos_add[1]), width=2 * ns_para_add[4],
                                                        height=2 * ns_para_add[6], angle=theta_add, theta1=0, theta2=90,
                                                        color='blue')
                        ped_ns_outfr_temp = patches.Arc(xy=(pos_add[0], pos_add[1]), width=2 * ns_para_add[4],
                                                        height=2 * ns_para_add[7], angle=theta_add, theta1=-90,
                                                        theta2=0,
                                                        color='blue')
                        ped_ns_outbl_temp = patches.Arc(xy=(pos_add[0], pos_add[1]), width=2 * ns_para_add[5],
                                                        height=2 * ns_para_add[6], angle=theta_add, theta1=90,
                                                        theta2=180,
                                                        color='blue')
                        ped_ns_outbr_temp = patches.Arc(xy=(pos_add[0], pos_add[1]), width=2 * ns_para_add[5],
                                                        height=2 * ns_para_add[7], angle=theta_add, theta1=180,
                                                        theta2=-90,
                                                        color='blue')

                        ax.add_artist(ped_circle_temp)
                        ax.add_artist(ped_num_temp)

                        ax.add_artist(ped_ns_outfl_temp)
                        ax.add_artist(ped_ns_outfr_temp)
                        ax.add_artist(ped_ns_outbl_temp)
                        ax.add_artist(ped_ns_outbr_temp)

                        dynamic_id_dict[id_add] = [ped_circle_temp, ped_num_temp, ped_ns_outfl_temp, ped_ns_outfr_temp,
                                                   ped_ns_outbl_temp, ped_ns_outbr_temp]

                    # update robot arrow
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)

                    # update time step
                    time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                def plot_value_heatmap():
                    pass

                def on_click(event):
                    anim.running ^= True
                    if anim.running:
                        anim.event_source.stop()
                        if hasattr(self.robot.policy, 'action_values'):
                            plot_value_heatmap()
                    else:
                        anim.event_source.start()

                fig.canvas.mpl_connect('key_press_event', on_click)

                anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
                anim.running = True

                if output_file is not None:
                    # ffmpeg_writer = animation.writers['ffmpeg']
                    # writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                    # anim.save(output_file, writer=writer)
                    anim.save('test.gif', writer='pillow')
                else:
                    plt.show()
            else:
                raise NotImplementedError

        if self.realpolicy is None:
            static_env_render()
        else:
            real_env_render()


