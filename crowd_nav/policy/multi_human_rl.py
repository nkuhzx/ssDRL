import torch
import numpy as np
import math

from numpy.linalg import norm
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human

from crowd_sim.envs.utils.utils import hh_intersection_area,point_to_segment_dist,hr_intersection_area_backup

class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)

                else:
                    human_states_for_pred_tensor=self.varied_pred_input_deal(state.human_states)
                    next_human_states_tensor=self.state_predictor(human_states_for_pred_tensor)
                    next_human_states_numpy=next_human_states_tensor.squeeze(0).data.cpu().numpy()

                    reward,humans_hr_social_stress, _ = self.estimate_reward(state.self_state,state.human_states,self.env.global_time,action)
                    next_human_states=self.transform_next_human_states(state.human_states,next_human_states_numpy,humans_hr_social_stress)

                rotated_batch_input=self.varied_input_deal(next_self_state,next_human_states)

                ##
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)

                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)
            self.last_human_states = self.transform_human(state.human_states)

        return max_action

    def configure_for_estimate_reward(self):

        self.time_limit = self.env.time_limit
        self.success_reward=self.env.success_reward
        self.collision_penalty=self.env.collision_penalty

        # construct the virtual robot
        self.robot_goal = [self.env.robot.gx, self.env.robot.gy]
        self.temp_robot=Robot(self.env.config,'robot')
        self.temp_robot.set(0,0,self.robot_goal[0],self.robot_goal[1],
                       0,0,0)
        self.temp_robot.stress_index=0
        self.temp_robot.hr_social_stress=0
        self.temp_robot.kinematics=self.kinematics

        # construct the virtual human instance
        self.temp_humans_max=[]
        for i in range(self.deal_human_num):
            temp_human=Human(self.env.config,"humans")
            temp_human.set(0,0,0,0,0,0,0)
            temp_human.hr_social_stress=0
            self.temp_humans_max.append(temp_human)

    def estimate_reward(self,self_state,human_states,global_time,action):

        # set the state for virtual robot
        self.temp_robot.set(self_state.px,self_state.py,self.robot_goal[0],self.robot_goal[1],
                       self_state.vx,self_state.vy,math.atan2(self_state.vy,self_state.vx))
        self.temp_robot.stress_index=0
        self.temp_robot.hr_social_stress=0

        # set the state for virtual humans
        human_num=len(human_states)
        temp_humans=self.temp_humans_max[:human_num]
        for i,human_state in enumerate(human_states,0):
            temp_humans[i].set(human_state.px,human_state.py,0,0,human_state.vx,human_state.vy,
                           math.atan2(human_state.vy,human_state.vx))
            temp_humans[i].hr_social_stress=0

        # get the parameters of tension space
        for human in temp_humans:
            human.get_nervous_space()

        self.temp_robot.get_nervous_space()

        # calculate the squeeze between two person
        squeeze_table = np.zeros((human_num, human_num))
        squeeze_table_test = np.zeros((human_num, human_num + 1))

        for i in range(human_num):
            for j in range(i + 1, human_num):
                squeeze_index = hh_intersection_area(temp_humans[i], temp_humans[j], 0.3)
                squeeze_table[i, j] = squeeze_index
                squeeze_table[j, i] = squeeze_index

                squeeze_table_test[i, j] = squeeze_index
                squeeze_table_test[j, i] = squeeze_index

            squeeze_table_test[i, -1] = temp_humans[i].squeeze_area

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(temp_humans):
            px = human.px - self.temp_robot.px
            py = human.py - self.temp_robot.py
            if self.temp_robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy

            else:
                vx = human.vx - action.v * np.cos(action.r + self.temp_robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.temp_robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.temp_robot.radius

            collision_index = hr_intersection_area_backup(human, self.temp_robot, 2)

            if closest_dist < 0 or collision_index:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(temp_humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = temp_humans[i].px - temp_humans[j].px
                dy = temp_humans[i].py - temp_humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - temp_humans[i].radius - temp_humans[j].radius
                if dist < 0:
                    pass

       # check if reaching the goal
        end_position = np.array(self.temp_robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.temp_robot.get_goal_position())) < self.temp_robot.radius

        last_pos = norm(np.array(self.temp_robot.get_position()) - np.array(self.temp_robot.get_goal_position()))
        now_pos = norm(end_position - np.array(self.temp_robot.get_goal_position()))
        distance_reward = last_pos - now_pos

        # calculate_human_social_stress
        for i, temp_human in enumerate(temp_humans):
            temp_human.set_human_social_stress()

        for i,temp_human in enumerate(temp_humans):
            if temp_human.squeeze_area != False:
                squeeze_index=list(np.where(squeeze_table[i]!=False)[0])
                squeeze_num=len(squeeze_index)
                ho_weight=[0 for num in range(squeeze_num)]
                for j in range(squeeze_num):
                    ho_weight[j]=temp_human.pos.distance(temp_humans[squeeze_index[j]].pos)
                ho_weight.append(temp_human.pos.distance(self.temp_robot.pos))
                ho_weight = np.array(ho_weight)
                ho_weight = 1 / ho_weight
                ho_weight=ho_weight/ho_weight.sum()
                hr_social_stress=temp_human.stress_index*ho_weight[-1]
            else:
                hr_social_stress=temp_human.stress_index

            temp_human.set_hr_social_stress(hr_social_stress)

        hr_social_stress_list = [0 for i in range(len(temp_humans))]
        hr_weight_list=[0 for i in range(len(temp_humans))]
        for i, temp_human in enumerate(temp_humans):

            hr_social_stress_list[i] = temp_human.hr_social_stress
            hr_weight_list[i]=temp_human.pos.distance(self.temp_robot.pos)

        hr_weight_list=np.array(hr_weight_list)
        hr_weight_list=1/hr_weight_list
        hr_weight_list=(hr_weight_list/hr_weight_list.sum())

        self.temp_robot.set_robot_composite_stress((np.multiply(hr_weight_list,np.array(hr_social_stress_list))).sum())

        if global_time >= self.time_limit - 1:
            reward = 0
            done = True
        elif collision:
            reward = self.collision_penalty
            done = True
        elif reaching_goal:
            reward = self.success_reward
            done = True
        else:
            reward = -0.5*self.temp_robot.hr_social_stress
            done = False

        hr_social_stress_dict= {}
        for i ,human in enumerate(temp_humans):

            hr_social_stress_dict[i]=human.hr_social_stress

        return reward,hr_social_stress_dict,done


    def transform_next_human_states(self,human_states,pred_human_states_numpy,hr_social_stress_dict):

        next_human_states=[]
        for i,human_state in enumerate(human_states,0):

            next_px,next_py,next_vx,next_vy=pred_human_states_numpy[i]

            next_human_state=ObservableState(next_px,next_py,next_vx,next_vy,human_state.radius,hr_social_stress_dict[i])
            next_human_states.append(next_human_state)

        return next_human_states

    def transform_human(self, human_state):

        state_tensor=self.varied_pred_input_deal(human_state,unsqueeze=False)

        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(human_state)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps], dim=1)

        return state_tensor


    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        #print(len(state.self_state))

        state_tensor=self.varied_input_deal(state.self_state,state.human_states,unsqueeze=False)

        # state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
        #                           for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps], dim=1)
        # else:
        #     state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)


    def varied_pred_input_deal(self,human_states,unsqueeze=True):

        # input px py vx vy && mask (0 for padding and 1 for real person)
        padding_num=self.deal_human_num-len(human_states)
        padding_tensor=torch.zeros(padding_num,5)

        if len(human_states)==0:
            human_states_tensor=padding_tensor

        else:
            human_states_tensor = torch.cat([torch.Tensor([[human_state.px,human_state.py,human_state.vx,human_state.vy,1]])
                                  for human_state in human_states], dim=0)

            human_states_tensor = torch.cat([human_states_tensor,padding_tensor],dim=0)

        if unsqueeze:

            human_states_tensor=human_states_tensor.unsqueeze(0)

        human_states_tensor=human_states_tensor.to(self.device)

        return human_states_tensor


    def varied_input_deal(self,self_state,human_states,unsqueeze=True):
        # Deal with human num==0
        #self_state
        self_state_=torch.Tensor(self_state.to_list()).to(self.device)

        fake_px,fake_py=self_state.px,self_state.py
        fake_vx,fake_vy=0,0
        fake_radius=0
        fake_hr_social_stress=0
        fake_state=ObservableState(fake_px,fake_py,fake_vx,fake_vy,fake_radius,fake_hr_social_stress)


        if len(human_states) == 0:

            fake_states=[fake_state for _ in range(self.deal_human_num)]

            batch_states = torch.cat([torch.Tensor([self_state + fake_state]).to(self.device)
                                           for fake_state in fake_states], dim=0)

            rotated_batch_input = self.rotate(batch_states)

            states_attribute=torch.zeros((self.deal_human_num)).unsqueeze(1).to(self.device)

            rotated_batch_input=torch.cat([rotated_batch_input,states_attribute],dim=1)



        # Deal with human num<5 ,padding
        elif len(human_states) <= self.deal_human_num:

            batch_states = torch.cat([torch.Tensor([self_state + human_state]).to(self.device)
                                       for human_state in human_states], dim=0)


            fake_states=[fake_state for _ in range(self.deal_human_num-len(human_states))]

            if(len(fake_states)!=0):

                padding=torch.cat([torch.Tensor([self_state + fake_state]).to(self.device)
                                               for fake_state in fake_states], dim=0)

                batch_states = torch.cat([batch_states, padding], dim=0)

                states_attribute = torch.cat([torch.ones((len(human_states))), torch.zeros((len(fake_states)))],dim=0)\
                    .unsqueeze(1).to(self.device)

            else:

                states_attribute = torch.ones((self.deal_human_num)).unsqueeze(1).to(self.device)

            rotated_batch_input = self.rotate(batch_states)

            rotated_batch_input=torch.cat([rotated_batch_input,states_attribute],dim=1)


        # Deal with human num<5, drop human state with distance
        elif len(human_states) > self.deal_human_num:
            distances = []
            for human_state in human_states:
                hr_distance = math.pow(human_state.px - self_state.px, 2) + math.pow(
                    human_state.py - self_state.py, 2)
                hr_distance = math.sqrt(hr_distance)
                distances.append(hr_distance)

            sort_indexs = list(np.argsort(distances))
            human_states_ = []
            for index in sort_indexs:
                if (index < self.deal_human_num):
                    human_states_.append(human_states[index])

            human_states = human_states_

            batch_states = torch.cat([torch.Tensor([self_state + human_state]).to(self.device)
                                           for human_state in human_states], dim=0)

            rotated_batch_input = self.rotate(batch_states)

            states_attribute=torch.ones((self.deal_human_num)).unsqueeze(1).to(self.device)

            rotated_batch_input=torch.cat([rotated_batch_input,states_attribute],dim=1)

        if unsqueeze:
            rotated_batch_input = rotated_batch_input.unsqueeze(0)

        return rotated_batch_input

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[2 * int(index)].append(1)
                            dm[2 * int(index) + 1].append(other_vx[i])
                            dm[2 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

