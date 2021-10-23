import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL
import math
from crowd_sim.envs.utils.state import ObservableState


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
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)

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

        return max_action

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal

        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

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

