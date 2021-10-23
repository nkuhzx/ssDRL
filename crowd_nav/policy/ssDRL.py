import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL



class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.input_dim=input_dim
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def set_device(self,device):
        self.device=device

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        state,mask=state[:,:,:self.input_dim],state[:,:,self.input_dim]
        size = state.shape
        #print(state)


        #equal self_state = state[0, 0, :self.self_state_dim]
        self_state = state[:, 0, :self.self_state_dim]

        #transfer to hr_social ratio
        hr_social_stress=state[:,:,-1].view(size[0], size[1], 1).squeeze(dim=2)

        hr_social_exp=torch.exp(hr_social_stress).float()
        hr_social_weight=(hr_social_exp/torch.sum(hr_social_exp, dim=1, keepdim=True)).unsqueeze(2)

        mlp1_output = self.mlp1(state.view((-1, size[2])))
        #print(mlp1_output.shape)


        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            #original_state = mlp1_output.view(size[0], size[1], -1)
            #original method#
            mask_=mask.unsqueeze(2).to(self.device)
            mask_ = mask_.expand((size[0], size[1], mlp1_output.shape[1])).contiguous()
            mask_weight=mask_/torch.sum(mask_,dim=1,keepdim=True)
            mask_weight[mask_weight != mask_weight] = 0
            global_state=torch.mul(mask_weight,mlp1_output.view(size[0], size[1], -1))
            global_state=torch.sum(global_state,dim=1,keepdim=True)

            # global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            #social_stress method#
            #global_state= torch.sum(torch.mul(hr_social_weight, original_state),dim=1,keepdim=True)

            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)


            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)

        scores=scores*mask.float()

        scores_exp = torch.exp(scores) * (scores != 0).float()

        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        weights[weights!=weights]=0

        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)

        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)

        value = self.mlp3(joint_state)
        if torch.sum(value!=value,dim=0).squeeze()>0:
            print(state)
            print(mask)

        return value


class ssDRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights

    def set_device(self, device):
        self.device = device
        self.model.to(device)
        self.model.set_device(device)

    def predict(self, state):


        def dist(human):
            # sort human order by decreasing distance to the robot
            return (-human.hr_social_stress, np.linalg.norm(np.array(human.position) - np.array(state.self_state.position)))
            #return (np.linalg.norm(np.array(human.position) - np.array(state.self_state.position)))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)

        return super().predict(state)
