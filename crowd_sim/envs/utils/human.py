from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState,ObservableState
import math
import numpy as np
import random
from crowd_sim.envs.utils.utils import Vector

STOPPING=0
WAITTING=1
WAITTING_TO_WALKING=2
WALKING=3

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

        # eye contact parameters and state
        eye_contact_parm=[float(x) for x in config.get(section, 'eye_contact_parm').split(', ')]
        self.eye_contact_parm=[math.radians(eye_contact_parm[0]),eye_contact_parm[1]]
        self.eye_contact_prob_threshold=config.getfloat(section,'eye_contact_prob_threshold')

        self.eye_contact = None
        self.eye_contact_interest_index=None

        # intention state (stopping, waiting, waiting_to_walking, walking)
        self.waiting_time_step_parm=[int(x) for x in config.get(section, 'waiting_time_step_parm').split(', ')]
        self.waiting_prob_threshold=config.getfloat(section,'waiting_prob_threshold')

        self.v_pref_backup=None
        self.waiting_time_range=None
        self.waiting_to_walk_count=None
        self.intention=None

    def sample_random_behavior_attributes(self,behavior_attributes=False,exist_waiting=False):

        self.v_pref_backup=self.v_pref

        self.eye_contact = False
        self.eye_contact_interest_index = 0

        self.waiting_time_range = [0, 0, 0]
        self.intention = WALKING
        self.waiting_to_walk_count = 0

        if behavior_attributes:

            self.eye_contact_interest_index=random.uniform(0.5,1)

            if exist_waiting and random.random()<self.waiting_prob_threshold:

                waiting_time_start=random.choice(range(self.waiting_time_step_parm[0])) * self.time_step
                waiting_time_step=random.choice(range(self.waiting_time_step_parm[1],self.waiting_time_step_parm[2]))*self.time_step
                self.waiting_time_range=[waiting_time_start,waiting_time_start+waiting_time_step,waiting_time_start+2*waiting_time_step]
                print(self.waiting_time_range)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        if self.intention==WAITTING:
            self.v_pref=0
        elif self.intention==WAITTING_TO_WALKING:
            self.v_pref=self.v_pref_backup*(1-math.exp(-math.pow(self.waiting_to_walk_count,2)/(2*15)))
        else:
            self.v_pref=self.v_pref_backup if self.v_pref!=self.v_pref_backup else self.v_pref
        action = self.policy.predict(state)
        return action

    def set_eye_contact_state(self,robot_ob:ObservableState):

        # the vector form human to robot
        hr_vector=Vector(robot_ob.px-self.px,robot_ob.py-self.py)
        vec_vector=Vector(self.vx,self.vy)

        if vec_vector.norm()==0:
            hrvec_cosine=hr_vector*vec_vector
        else:
            hrvec_cosine=hr_vector*vec_vector/(hr_vector.norm()*vec_vector.norm())

        hrvec_angle=math.acos(hrvec_cosine)


        hr_distance=hr_vector.norm()

        eye_contact_fov,eye_contact_dist=self.eye_contact_parm

        # the prob about eye contact is increase with the dist/fov decrease
        prob=math.exp(-1/2*(math.pow((hrvec_angle/eye_contact_fov),2)+math.pow((hr_distance/eye_contact_dist),2)))

        prob_with_interest=prob*self.eye_contact_interest_index
        self.eye_contact=True if prob_with_interest>self.eye_contact_prob_threshold else False


    def set_intention_state(self,global_time):


        if self.reached_destination():
            self.intention=STOPPING
            self.waiting_to_walk_count=0
        elif global_time>=self.waiting_time_range[0] and global_time<self.waiting_time_range[1]:
            self.intention=WAITTING
            self.waiting_to_walk_count = 0
        elif global_time>=self.waiting_time_range[1] and global_time<=self.waiting_time_range[2]:
            self.intention=WAITTING_TO_WALKING
            self.waiting_to_walk_count +=1
        else:
            self.intention=WALKING
            self.waiting_to_walk_count = 0


    def get_nervous_space(self,action=None):

        #k1,k2,k3,k4,k5,k6=0.6,0.3,0.1,0.5,0.2,0.1
        #for real scene
        k1, k2, k3= self.tension_parm[0], self.tension_parm[1], self.tension_parm[2]
        #aout0=bout0=np.array([0.45,0.45])
        #for real scene
        aout0 = bout0 = np.array(self.tension_space_out)

        k4, k5, k6 = 0.5, 0.2, 0.1
        k_waiting,k_eye_contact=0.7,0.45

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

        if self.intention==WAITTING or self.intention==WAITTING_TO_WALKING:
            vr =k_waiting*self.v_pref_backup

        if self.eye_contact:
            self.aout=aout0+vr*np.array([k1,k2])*k_eye_contact
            self.bout = bout0 + vr * np.array([k3, k3])*k_eye_contact
        else:
            self.aout=aout0+vr*np.array([k1,k2])
            self.bout = bout0 + vr * np.array([k3, k3])

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

