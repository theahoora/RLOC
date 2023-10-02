from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

print(torch.__version__)

class AC:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.actor = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

        self.critic = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]+2), # Q , a ,tau
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001)

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # train the DNN
        criterion = nn.BCELoss()
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        self.optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

# class Critic(nn.Module):
#     def __init__(self,input_dim):
#         super(Critic,self).__init__()
        
#         self.l1 = nn.Linear(input_dim,256)
#         self.l2 = nn.Linear(256,64)
#         self.q_1 = nn.Linear(64,1)
#         self.l3 = nn.Linear(64+input_dim,64)
#         self.q_2 = nn.Linear(64,1)








# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + 2*self.net[-1] + 2))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

        self.critic = nn.Sequential(
            nn.Linear(self.net[0],256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        self.critic_opt = optim.Adam(self.critic.parameters(),lr=self.lr,weight_decay=0.0001)

    def remember(self, h, m_p,m,r,r_mean):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        # print('h',h.shape)
        # print('m',m.shape)
        # print('r',r.shape)
        self.memory[idx, :] = np.hstack((h,m_p, m,r,r_mean))

        self.memory_counter += 1

    def encode(self, h, m_pred,m,r,r_mean):
        # encoding the entry
        self.remember(h, m_pred,m,r,r_mean)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            # for i in range(3):
            self.learn()

        if self.memory_counter % (self.training_interval//2) == 0:
            self.train_critic()        
            # self.memory_counter = 1

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:self.net[0]+self.net[-1]])
        m_gen = torch.FloatTensor(batch_memory[:,self.net[0]+self.net[-1]:-2])
        r = torch.Tensor(batch_memory[:, -2:-1])/10000
        r_mean = torch.Tensor(batch_memory[:, -1:])/10000

        # train the DNN
        # criterion = nn.BCELoss()
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(h_train)

        prob = (m_gen*predict) + ((1-m_gen)*(1-predict))
        # print(prob>0.05)
        # mask = prob>0.1
        # prob = prob[mask]
         
        log_prob = torch.log(prob)
        log_prob_sum = torch.sum(log_prob,dim=-1)

        old_prob =((m_gen*m_train) + ((1-m_gen)*(1-m_train)))
        old_log_prob = torch.log(old_prob)
        old_log_prob_sum = torch.sum(old_log_prob,dim=-1)

        # prob_mul = torch.mul(prob.detach(),dim=-1)
        # old_prob_mul = torch.mul(old_prob,dim=-1)
        ratio = torch.prod(prob/old_prob,dim=-1)

        # print('rr',r,r_mean)
        cat = torch.distributions.Categorical(prob)
        # loss = -torch.mean(r*log_prob_sum * torch.exp(log_prob_sum.detach() - old_log_prob_sum))#criterion(predict, m_train)
        # loss = -torch.mean(torch.clip(r-r_mean,-50,50)*log_prob_sum)# - 0.1*cat.entropy().mean()#criterion(predict, m_train)
        loss = -torch.mean((r-r_mean)*ratio)
        # loss = -torch.mean((r-self.critic(h_train).detach())*ratio)
        # loss = -torch.mean(r*ratio)
        
        # print(loss)
        # print(torch.exp(log_prob_sum.detach() - old_log_prob_sum))
        # print(log_prob_sum.mean(),ratio.mean())
        # print(torch.log(loss))
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
        self.optimizer.step()
        
        self.cost = loss.item()
        # assert(self.cost > 0)
        self.cost_his.append(self.cost)
    
    def train_critic(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:self.net[0]+self.net[-1]])
        m_gen = torch.FloatTensor(batch_memory[:,self.net[0]+self.net[-1]:-2])
        r = torch.Tensor(batch_memory[:, -2:-1])/10000
        r_mean = torch.Tensor(batch_memory[:, -1:])/10000


        r_c = self.critic(h_train)
        
        loss = F.mse_loss(r_c,r)
        self.critic_opt.zero_grad()

        loss.backward()

        
        torch.nn.utils.clip_grad_value_(self.critic.parameters(),1)
        self.critic_opt.step()



    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy().squeeze()

        return m_pred

        # if mode is 'OP':
        #     return self.knm(m_pred[0], k)
        # elif mode is 'KNN':
        #     return self.knn(m_pred[0], k)
        # else:
        #     print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

