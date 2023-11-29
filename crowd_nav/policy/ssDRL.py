import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL



class StatePredictor(nn.Module):

    def __init__(self, input_dim, mlp1_dims, mlp2_dims, attention_dims, hmlp_dims,with_global_state):
        super().__init__()
        self.trainable=True
        self.input_dim=input_dim
        self.global_state_dim = mlp1_dims[-1]
        self.weightsum_state_dim=mlp2_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)

        self.attention_weights = None

        H_dim=mlp2_dims[-1]*2
        self.human_state_predictor=mlp(H_dim,hmlp_dims)


    def set_device(self,device):
        self.device=device

    def forward(self,state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        # bs x human_num x 5, bs x human_num x 1
        state,mask=state[:,:,:-1],state[:,:,-1]
        size = state.shape

        # bs x human_num , 100
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        #print(mlp1_output.shape)

        # bs x human_num , 50
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
        weighted_human_feature=torch.mul(weights,features)

        weighted_sum_feature = torch.sum(weighted_human_feature, dim=1).view(size[0],1,-1)

        weighted_sum_feature =weighted_sum_feature.expand((size[0],size[1], self.weightsum_state_dim)).contiguous()
        # concatenate agent's state with global weighted humans' state

        joint_feature = torch.cat([features, weighted_sum_feature], dim=2)

        next_human_state=self.human_state_predictor(joint_feature)

        return next_human_state

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
        emb_input_dims=config.getint('state_predictor','input_dims')
        pred_hmlp_dims=[int(x) for x in config.get('state_predictor', 'hmlp_dims').split(', ')]
        self.state_predictor=StatePredictor(emb_input_dims,mlp1_dims,mlp2_dims,attention_dims,pred_hmlp_dims,with_global_state)
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

        self.state_predictor.to(device)
        self.state_predictor.set_device(device)


    def get_state_dict(self):

        if self.state_predictor.trainable:
            return {
                'value_network': self.model.state_dict(),
                'state_predictor': self.state_predictor.state_dict()
            }
        else:
            return {
                'value_network': self.model.state_dict()
            }

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable:
            self.model.load_state_dict(state_dict['value_network'])
            self.state_predictor.load_state_dict(state_dict['state_predictor'])
        else:
            self.model.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)
