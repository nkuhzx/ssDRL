import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
import pandas as pd
import os
import math
import copy
import matplotlib.pyplot as plt
from matplotlib import animation

class Realscene(Policy):
    def __init__(self):
        super().__init__()

        self.trainable=False
        self.centralized=True
        self.kinematics = 'holonomic'
        self.time_step=None
        self.human_radius=None

        self.interval=None
        self.case_length=None

        self.fragment_num=None
        self.capacity=None

        self.frameNo_list=None

        #for show



    def configure(self, config):
        self.time_step = config.getfloat('realscene', 'time_step')

        self.human_radius=config.getfloat('humans', 'radius')
        # get dataset from txt
        relativePath=config.get('realscene','path')
        rootPath = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path=os.path.join(rootPath,relativePath)

        self.dataset=pd.read_csv(self.dataset_path,sep=',',header=None,names = ['frameNo','id', 'x','y'])
        self.frameNo_list=np.array(self.dataset['frameNo'].unique())

        # control the data interval
        self.interval=config.getint('realscene', 'interval')
        self.case_length=config.getint('realscene', 'case_length')

        self.capacity=math.floor((self.frameNo_list.shape[0]-self.case_length)/self.interval)+1

        return

    def get_scene_width(self):
        lim_sup = max(max(self.dataset['y']), max(self.dataset['y']))
        lim_low=abs(min(min(self.dataset['y']), min(self.dataset['y'])))
        scene_width=max(lim_sup,lim_low)
        return scene_width

    def get_human_id_list(self,case_counter,frame_step=None):
        case_frameNo=self.frameNo_list[case_counter*self.interval:case_counter*self.interval+self.case_length]

        if frame_step is None:
            current_case = self.dataset.loc[(self.dataset['frameNo'] >= case_frameNo.min()) & (self.dataset['frameNo'] <= case_frameNo.max())]
            id_unreval = np.array(current_case['id'].unique())
        elif frame_step<self.case_length:
            current_frame = self.dataset.loc[self.dataset['frameNo'] == case_frameNo[frame_step]]
            id_unreval=np.array(current_frame['id'].unique())
        else:
            raise Exception('frame_step is out of the case length: {}/{}'.format(frame_step,self.case_length))
        return id_unreval.tolist()

    def get_human_pos(self,id,case_counter,fram_step):


        case_frameNo=self.frameNo_list[case_counter*self.interval:case_counter*self.interval+self.case_length]
        frameNo=case_frameNo[fram_step]
        if isinstance(id,list):

            if len(id)==0:
                position=list()
            elif len(id)==1:
                position = np.array(self.dataset.loc[(self.dataset['id'] .isin(id)) & (self.dataset['frameNo'] == frameNo), ['x','y']]).squeeze()
                position=[position.tolist()]


            else:
                predeal=self.dataset.loc[(self.dataset['id'].isin(id)) & (self.dataset['frameNo'] == frameNo)].copy()

                # sort by id_list to generate a new dataframe
                predeal.loc[:,'id'] = predeal.loc[:,'id'].astype('category')

                predeal.loc[:,'id'].cat.set_categories(id,inplace=True)

                predeal.sort_values('id',inplace=True)

                position=np.array(predeal.loc[:,['x','y']]).squeeze()

                position = position.tolist()

        else:
            position=np.array(self.dataset.loc[(self.dataset['id'] == id)&(self.dataset['frameNo']==frameNo), ['x', 'y']]).squeeze()

        return position

    def predict(self, id,case_counter,fram_step):
        pass

        return

