from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState,ObservableState
import math
import numpy as np



class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.aout=None
        self.bout=None
        self.amid=None
        self.bmid=None

        self.afront=None
        self.aback=None
        self.bleft=None
        self.bright=None

        self.lastpara=None
        self.init=True

        self.squeeze_area=None

        self.original_area=None

        self.tension_space_out=[float(x) for x in config.get(section, 'tension_space_out').split(', ')]
        self.tension_parm=[float(x) for x in config.get(section, 'tension_space_parm').split(', ')]


    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)

        action = self.policy.predict(state)
        return action



    def get_nervous_space(self,action=None):

        #k1,k2,k3,k4,k5,k6=0.6,0.3,0.1,0.5,0.2,0.1
        #for real scene
        k1, k2, k3= self.tension_parm[0], self.tension_parm[1], self.tension_parm[2]
        #aout0=bout0=np.array([0.45,0.45])
        #for real scene
        aout0 = bout0 = np.array(self.tension_space_out)

        k4, k5, k6 = 0.5, 0.2, 0.1

        amid0=bmid0=np.array([0.4,0.4])

        if action is None:

            if self.vx is not None and self.vy is not None:
                vr_2=math.pow(self.vx,2)+math.pow(self.vy,2)
                vr=math.sqrt(vr_2)
            else:
                vr_2=self.v_pref**2
                vr=self.v_pref
        else:
            raise NotImplemented

        self.aout=aout0+vr*np.array([k1,k2])
        self.bout=bout0+vr*np.array([k3,k3])
        self.amid=amid0+vr*np.array([k4,k5])
        self.bmid = bmid0 + vr * np.array([k6, k6])


        if self.init==False:

            self.lastpara = [self.afront, self.aback, self.bleft, self.bright]

            self.original_area = [self.aout[0], self.aout[1], self.bout[0], self.bout[1]]


        self.afront=self.aout[0]
        self.aback=self.aout[1]
        self.bleft=self.bout[0]
        self.bright=self.bout[1]

        if self.init==True:

            self.lastpara=[self.afront,self.aback,self.bleft,self.bright]

            self.original_area=[self.aout[0],self.aout[1],self.bout[0],self.bout[1]]

            self.squeeze_area=False

            self.init=False

    def get_change_space(self):
        self.original_area=[self.afront,self.aback,self.bleft,self.bright]

    def set_nervous_space_para(self):

        k=0.4

        #self.lastpara = [self.aout[0], self.aback, self.bleft, self.bright]

        self.aout[0] = self.afront
        self.aout[1] = self.aback
        self.bout[0] = self.bleft
        self.bout[1] = self.bright

        self.amid[0] = self.radius + k * self.aout[0]

        self.amid[1] = self.radius + k * self.aout[1]
        self.bmid[0] = self.radius + k * self.bout[0]
        self.bmid[1] = self.radius + k * self.bout[1]


    def nervous_index(self,time):

        k=-3
        if self.squeeze_area:
            ns_area_original=0.25*math.pi*(self.original_area[0]+self.original_area[1])*(self.original_area[2]+self.original_area[3])
            ns_area = 0.25 * math.pi * (
                        self.afront * self.bleft + self.afront * self.bright + self.aback * self.bleft + self.aback * self.bright)
            ns_area_last = 0.25 * math.pi * (
                        self.lastpara[0] * self.lastpara[2] + self.lastpara[0] * self.lastpara[3] + self.lastpara[1] *
                        self.lastpara[2] + self.lastpara[1] * self.lastpara[3])
            #index_now=1/(1+math.exp(k*(ns_area/ns_area_original-0.5)))
            #index_last=1/(1+math.exp(k*(ns_area_last/ns_area_original-0.5)))
            changearea=1/(1+math.exp(k*(0.5-ns_area/ns_area_original)))
        else:
            changearea=0

        self.stress_index=changearea

        return changearea


    def set_human_social_stress(self):
        k=-3
        if self.squeeze_area:
            ns_area_original=0.25*math.pi*(self.original_area[0]+self.original_area[1])*(self.original_area[2]+self.original_area[3])
            ns_area = 0.25 * math.pi * (
                        self.afront * self.bleft + self.afront * self.bright + self.aback * self.bleft + self.aback * self.bright)
            ns_area_last = 0.25 * math.pi * (
                        self.lastpara[0] * self.lastpara[2] + self.lastpara[0] * self.lastpara[3] + self.lastpara[1] *
                        self.lastpara[2] + self.lastpara[1] * self.lastpara[3])
            #index_now=1/(1+math.exp(k*(ns_area/ns_area_original-0.5)))
            #index_last=1/(1+math.exp(k*(ns_area_last/ns_area_original-0.5)))
            changearea=1/(1+math.exp(k*(0.5-ns_area/ns_area_original)))
        else:
            changearea=0

        self.stress_index=changearea


    def set_hr_social_stress(self,value):
        self.hr_social_stress=value


    def get_observable_state(self):

        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius,self.hr_social_stress)

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
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius,self.hr_social_stress)

