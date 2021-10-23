from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
import numpy as np
import math


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.afront = None
        self.aback  = None
        self.bfront = None
        self.bback  = None


    def act(self, ob):

        # no observation in sensor range
        if len(ob)==0:

            velocity=np.array((self.gx-self.px,self.gy-self.py))
            speed = np.linalg.norm(velocity)
            pref_vel=velocity/speed if speed>self.v_pref else velocity
            action=ActionXY(pref_vel[0],pref_vel[1])


        # normal sense
        else:
            if self.policy is None:
                raise AttributeError('Policy attribute has to be set!')
            state = JointState(self.get_full_state(), ob)
            action = self.policy.predict(state)

        return action


    def get_nervous_space(self,action=None):

        amax=0.15
        k=0.5

        if action is None:
            if self.vx is not None and self.vy is not None:
                vr_2=math.pow(self.vx,2)+math.pow(self.vy,2)
                vr=math.sqrt(vr_2)
            else:
                vr_2=self.v_pref**2
                vr=self.v_pref
        else:
            vr_2 = math.pow(action.vx, 2) + math.pow(action.vy, 2)
            vr = math.sqrt(vr_2)


        self.afront=self.radius+amax*vr_2
        self.aback=self.radius
        self.bfront=self.radius+k*vr
        self.bback=self.bfront

        return(self.afront,self.aback,self.bfront,self.bback)

    def set_robot_composite_stress(self,value):
        self.hr_social_stress=value


