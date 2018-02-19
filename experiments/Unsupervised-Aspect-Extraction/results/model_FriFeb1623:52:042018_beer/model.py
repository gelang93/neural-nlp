import math
import pickle
import os, shutil

import numpy as np

import torch
from torch import optim

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

file_name = os.path.abspath(__file__)


class EmbedSentence(nn.Module) :
    def __init__(self, vocab_size, embed_size, n_aspects, pre_embed=None) :
        super(EmbedSentence, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        if pre_embed is not None :
            self.embedding.weight.data.copy_(torch.from_numpy(pre_embed))

        for param in self.embedding.parameters() :
            param.requires_grad = False
            
        self.M = nn.Linear(embed_size, embed_size, bias=False)
        
        self.recon = nn.Linear(embed_size, n_aspects)
        
        self.T = nn.Linear(n_aspects, embed_size, bias=False)
        
    def forward(self, sentence, imask, reconstruct=False) :
        #(B, L), #(B, L)
        embedding = self.embedding(sentence) #(B, L, E)

        negmask = (1 - imask).unsqueeze(-1).float()
        masked_embedding = embedding * negmask
        avg_embedding = torch.sum(masked_embedding, dim=1) / torch.sum(negmask, dim=1) #(B, E)

        y = self.M(avg_embedding) #(B, E)
        attn = torch.bmm(embedding, y.unsqueeze(-1)) #(B, L, 1)
        attn = attn.masked_fill_(imask.byte().unsqueeze(-1), -float('inf')) # (B, L, 1)
        attn_weights = F.softmax(attn, dim=1) # (B, L, 1)
        
        mix = torch.bmm(attn_weights.transpose(1, 2), embedding) #(B, 1, E)
        mix = mix.squeeze(1) #(B, E)
        
        p_t, r_s = None, None
        if reconstruct :
            p_t = F.softmax(self.recon(mix), dim=-1) #(B, K)
            r_s = self.T(p_t) #(B, E)
            r_s = F.normalize(r_s, p=2, dim=1)
        
        z_s = F.normalize(mix, p=2, dim=1)
        return z_s, attn_weights, p_t, r_s
    
    def get_reg(self) :
        weight = F.normalize(self.T.weight, dim=0)
        weight = torch.mm(weight.transpose(0, 1), weight)
        weight = weight + -1.0*Variable(torch.eye(self.T.weight.size()[1]).cuda())
        return weight.norm()

class Model() :
    def __init__(self, **kwargs) :
        self.lang_1 = kwargs['lang_1']
        self.model = EmbedSentence(self.lang_1['vocab_size'], self.lang_1['word_dim'], kwargs['num_aspect'], pre_embed=self.lang_1['embeddings'])
        self.model.cuda()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))

        import time
        self.time_str = time.ctime().replace(' ', '')
        self.dirname = 'results/model_'

    def prepare_data(self, input_batch) :
        maxlen = max([len(x) for x in input_batch])
        input_masks = np.zeros((len(input_batch), maxlen))
        input_lengths = []

        for i, x in enumerate(input_batch) :
            rem = maxlen - len(x)
            input_lengths.append(len(x))
            input_masks[i, len(x):] = 1  
            input_batch[i] = [int(y) for y in x] + [0]*rem    

        input_variable = Variable(torch.LongTensor(input_batch).cuda())
        input_masks = Variable(torch.from_numpy(input_masks).cuda())

        return input_variable, input_masks

    def train_batch(self, input_batch, negative_batch) :
        model = self.model
        optimizer = self.optimizer

        optimizer.zero_grad()
        si, smask = self.prepare_data(input_batch) #(B, L), #(B, L)
        mix_s, attn_weights, p_t, r_s = model(si, smask, reconstruct=True) #(B, E), _, (B, K), (B, E)
        pos_dot = torch.bmm(mix_s.unsqueeze(1), r_s.unsqueeze(-1)).squeeze()
        ranking_loss = torch.nn.MarginRankingLoss(margin=1)
        loss = 0
        y = Variable(torch.ones(si.size()[0]).cuda())

        for n in range(len(negative_batch)) :
            ni, nmask = self.prepare_data(negative_batch[n])
            mix_n, _,_,_ = model(ni, nmask)
            neg_dot = torch.bmm(mix_n.unsqueeze(1), r_s.unsqueeze(-1)).squeeze()
            loss += ranking_loss(pos_dot, neg_dot, y)
            
        loss_reg = model.get_reg()
        total_loss = loss + 1.0 * loss_reg
        total_loss.backward()
        optimizer.step()
        
        self.loss = loss.data[0]
        self.loss_reg = loss_reg.data[0]

    def save_values(self, add_name='') :
        dirname = self.dirname + self.time_str + add_name
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        torch.save(self.model.state_dict(), dirname + '/encoder.th')
        torch.save(self.optimizer.state_dict(), dirname + '/encoder_optimizer.th')

        return dirname

    def load_values(self, dirname) :
        self.model.load_state_dict(torch.load(dirname + '/encoder.th'))
        self.optimizer.load_state_dict(torch.load(dirname + '/encoder_optimizer.th'))
