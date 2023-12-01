import logging
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model,state_predictor, memory, device, batch_size,human_num):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.state_predictor=state_predictor
        self.device = device
        self.criterion=nn.MSELoss().to(device)
        self.criterion_sp = nn.MSELoss(reduction="none").to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.state_predictor_update_interval=human_num


    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        if self.state_predictor.trainable:
            self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate,momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        average_epoch_s_loss=0
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_s_loss = 0
            update_counter = 0
            for data in self.data_loader:
                inputs, values,human_states_inputs,next_human_states_gt,final_mask = data
                inputs = Variable(inputs)
                values = Variable(values)
                inputs=inputs.to(self.device)
                values=values.to(self.device)

                human_states_inputs = Variable(human_states_inputs)
                next_human_states_gt = Variable(next_human_states_gt)
                human_states_inputs=human_states_inputs.to(self.device)
                next_human_states_gt=next_human_states_gt.to(self.device)

                final_mask=Variable(final_mask)
                final_mask=final_mask.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        next_human_states_pred=self.state_predictor(human_states_inputs)
                        s_loss = self.criterion_sp(next_human_states_pred[:,1:,:], next_human_states_gt[:, 1:, :-1])
                        s_loss = torch.mean(s_loss,dim=1)
                        s_loss = torch.mul(s_loss,final_mask)
                        if torch.sum(final_mask)==0:
                            s_loss = torch.sum(s_loss)
                        else:
                            s_loss = torch.sum(s_loss)/torch.sum(final_mask)
                        s_loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss += s_loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            average_epoch_s_loss = epoch_s_loss/ len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E, %.2E', epoch,average_epoch_loss,average_epoch_s_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        s_losses=0
        for _ in range(num_batches):
            inputs,values,human_states_inputs,next_human_states_gt,final_mask = next(iter(self.data_loader))

            inputs = Variable(inputs)
            values = Variable(values)

            human_states_inputs = Variable(human_states_inputs)
            next_human_states_gt= Variable(next_human_states_gt)
            final_mask=Variable(final_mask)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

            if self.state_predictor.trainable:
                update_state_predictor = True

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    next_human_states_pred = self.state_predictor(human_states_inputs)
                    s_loss = self.criterion_sp(next_human_states_pred[:,1:,:], next_human_states_gt[:, 1:, :-1])
                    s_loss = torch.mean(s_loss, dim=1)
                    s_loss = torch.mul(s_loss, final_mask)
                    if torch.sum(final_mask) == 0:
                        s_loss = torch.sum(s_loss)
                    else:
                        s_loss = torch.sum(s_loss) / torch.sum(final_mask)
                    s_loss.backward()
                    self.s_optimizer.step()
                    s_losses += s_loss.data.item()


        average_loss = losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_loss, average_s_loss)

        return average_loss,average_s_loss
