import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss
import json
import os


class MF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MF, self).__init__(conf, training_set, test_set)
        # 如果存在。pt 则加载
        file = self.config.config['training.set']
        self.parent_dir = os.path.dirname(file)
        if os.path.exists(os.path.join(self.parent_dir, 'user_emb.pt')):
            self.load_pt()
        else:
            self.data.user_emb = None
            self.data.item_emb = None
        self.model = Matrix_Factorization(self.data, self.emb_size)

        if os.path.exists('./dataset/our/title_tensor.pt'):
            self.tittle_emb = torch.load('./dataset/our/title_tensor.pt').cuda()
        else:
            self.tittle_emb = None



    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                if self.tittle_emb == None:
                    tittle_emb = None
                else:
                    neg_tittle_emb, pos_tittle_emb = self.tittle_emb[neg_idx], self.tittle_emb[pos_idx]
                    tittle_emb = [neg_tittle_emb, pos_tittle_emb]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb, tittle_emb) + l2_reg_loss(self.reg,
                                                                                                      user_emb,
                                                                                                      pos_item_emb,
                                                                                                      neg_item_emb) / self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
            if batch_loss.item() < 0.001:
                break
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        self.save2()

    def save2(self):
        file = self.config.config['training.set']
        parent_dir = os.path.dirname(file)

        torch.save(self.best_user_emb, os.path.join(parent_dir, 'user_emb.pt'))
        torch.save(self.best_item_emb, os.path.join(parent_dir, 'item_emb.pt'))
        with open(os.path.join(parent_dir, 'user.json'), 'w') as f:
            json.dump(self.data.user, f)
        with open(os.path.join(parent_dir, 'item.json'), 'w') as f:
            json.dump(self.data.item, f)

    def load_pt(self):
        file = self.config.config['training.set']
        parent_dir = os.path.dirname(file)

        # self.data.user_emb = torch.load(os.path.join(parent_dir, 'user_emb.pt'))
        # self.data.item_emb = torch.load(os.path.join(parent_dir, 'item_emb.pt'))
        self.data.user_emb = None
        self.data.item_emb = None

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            if self.tittle_emb == None:
                score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            else:
                score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
                + torch.matmul(self.user_emb[u], self.tittle_emb.transpose(0, 1))

            return score.cpu().numpy()


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        if self.data.user_emb != None:
            embedding_dict['user_emb'].data[:self.data.user_emb.shape[0]] = self.data.user_emb
            embedding_dict['item_emb'].data[:self.data.item_emb.shape[0]] = self.data.item_emb

        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']
