import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel
import random
import numpy as np

class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1)

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none':
          continue
        fn = self._get_activation(name)
        unweighted = states + c * (fn(h) - states)
        # print('unweighted',unweighted.shape)
        s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)#按列求和
        # print('probs[offset:offset+i+1, k]',probs[offset:offset+i+1, k])
        # print('probs[offset:offset+i+1, k].unsqueeze(-1)',probs[offset:offset+i+1, k].unsqueeze(-1).shape)
        # print('probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1)',probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1).shape)
        # # print('torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted',probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted.shape)
        # print('torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)',torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0).shape)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()
        self.n = 0
    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      k = sum(i for i in range(1, STEPS+1))
      weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)
      self.weights = Variable(weights_data.cuda(), requires_grad=True)
      self._arch_parameters = [self.weights]
      for rnn in self.rnns:
        rnn.weights = self.weights

    def arch_parameters(self):
      return self._arch_parameters

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def genotype(self):

      def _parse(probs):
        epsilon = 1
        num = random.random()
        if num < epsilon:
          print('epsilon-----------------------------------------------------------',self.n)
          self.n+=1
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          if num > epsilon:
            j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
          else:
            # print('epsilon-----------------------------------------------------')
            edge = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))
            # print(edge)
            if 2>len(edge)>1:
              j = np.random.choice(edge[1])
            elif 3>len(edge)>2:
              j = np.random.choice(edge[1:])
            elif 3>len(edge)>2:
              j = np.random.choice(edge[1:3])
            else:
              j = np.random.choice(edge)

          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          if num > epsilon:
            gene.append((PRIMITIVES[k_best], j))
          else:
            ##########################################随机
            # num_ops = len(PRIMITIVES)
            # opsss = [i for i in range(num_ops)]
            # opss = opsss[1:]
            # k_best = np.random.choice(opss)
            # gene.append((PRIMITIVES[k_best], j))
            ############################################
            W_ops = W[j].tolist()
            Ws = [i for i in W[j].tolist()]
            W = W_ops
            W_ = W.pop(PRIMITIVES.index('none'))
            ops = sorted(W)
            k = np.random.choice(ops[-2:])
            k_best = Ws.index(k)
            gene.append((PRIMITIVES[k_best], j))

          start = end
        return gene

      gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
      genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
      return genotype
