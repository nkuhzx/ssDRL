import numpy as np
from numpy.linalg import norm
import abc
import logging
import math
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.utils import Point


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        sensing_range= config.get(section, 'sensing_range')
        if sensing_range.isdigit():
            self.sensing_range=float(sensing_range)
        else:
            self.sensing_range=None
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

        # reback to last state
        self.last_px=None
        self.last_py=None
        self.last_vx=None
        self.last_vy=None
        self.last_theta=None

        self.pos=None
        self.angle=None

        self.stress_index=None
        self.hr_social_stress=None

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

        self.last_px=px
        self.last_py=py
        self.last_vx=vx
        self.last_vy=vy
        self.last_theta=theta

        self.pos=Point(px,py)
        self.angle=math.degrees(math.atan2(vy,vx))

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):

        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):

        if isinstance(action,list):
            assert len(action)==2
            next_px,next_py=action
            if self.kinematics == 'holonomic':
                next_vx = (next_px-self.px)/self.time_step
                next_vy = (next_py-self.py)/self.time_step
            else:
                raise NotImplemented

        else:
            self.check_validity(action)
            pos = self.compute_position(action, self.time_step)
            next_px, next_py = pos
            if self.kinematics == 'holonomic':
                next_vx = action.vx
                next_vy = action.vy
            else:
                next_theta = self.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,self.hr_social_stress)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        if isinstance(action,list):
            assert len(action)==2
            if self.kinematics == 'holonomic':
                self.vx = (action[0] - self.px) / self.time_step
                self.vy = (action[1] - self.py) / self.time_step
            else:
                raise NotImplemented

            self.px, self.py = action[0], action[1]

            self.pos = Point(self.px, self.py)
            self.angle = math.degrees(math.atan2(self.vy, self.vx))

        else:


            self.check_validity(action)
            pos = self.compute_position(action, self.time_step)
            self.px, self.py = pos
            if self.kinematics == 'holonomic':
                self.vx = action.vx
                self.vy = action.vy
            else:
                self.theta = (self.theta + action.r) % (2 * np.pi)
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)

            self.pos=Point(self.px,self.py)
            self.angle=math.degrees(math.atan2(self.vy,self.vx))



    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

