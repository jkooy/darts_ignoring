""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import utils
import os
import torch.nn as nn
from config import SearchConfig
from torch.autograd import Variable

config = SearchConfig()


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim, model, Likelihood, batch_size, step):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        sloss = self.v_net.loss(trn_X, trn_y) # L_trn(w)
        
        
        dataIndex = len(trn_y)+step*batch_size
        ignore_crit = nn.CrossEntropyLoss(reduction='none').cuda()       
        # forward
        logits = self.v_net(trn_X)
        
        '''
        loss = torch.dot(Likelihood[step*batch_size:dataIndex], ignore_crit(logits, trn_y))
#         loss = torch.dot(Likelihood[step*batch_size:dataIndex], ignore_crit(logits, trn_y))/(Likelihood[step*batch_size:dataIndex].sum())
        '''
        # sigmoid loss
        loss = torch.dot(torch.sigmoid(Likelihood[step*batch_size:dataIndex]), ignore_crit(logits, trn_y))/(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        
        
        # compute gradient
        gradients = torch.autograd.grad(loss, self.v_net.weights(), retain_graph=True)
        dtloss_ll = torch.autograd.grad(loss, Likelihood)
        print('dtloss_ll', sum(dtloss_ll).sum())
        
        
        dtloss_w = []
        # do virtual step (update gradient)       
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum    
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w)) 
                dtloss_w.append(m + g + self.w_weight_decay*w)
                
            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a) 
                
        return dtloss_w, dtloss_ll
        # 1399:[48, 3, 3, 3], 1:25000
        
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, model, likelihood, Likelihood_optim, batch_size, step):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        dtloss_w, dtloss_ll = self.virtual_step(trn_X, trn_y, xi, w_optim, model, likelihood, batch_size, step)
        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
#         v_grads = torch.autograd.grad(loss, v_weights, retain_graph=True)
    
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights, retain_graph=True)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        
        
        ########## tuple

        dvloss_tloss = 0  
        for dv, dt in zip(dw, dtloss_w):
            grad_valw_d_trainw = torch.div(dv, dt)
            grad_valw_d_trainw[torch.isinf(grad_valw_d_trainw)] = 0
            grad_valw_d_trainw[torch.isnan(grad_valw_d_trainw)] = 0
            grad_val_train = torch.sum(grad_valw_d_trainw)
            dvloss_tloss += grad_val_train
            
        
        dlikelihood = dvloss_tloss* dtloss_ll[0]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h
        Likelihood_optim.zero_grad()
        likelihood.grad = dlikelihood
        print('likelihood gradient is:', likelihood.grad.sum())
        Likelihood_optim.step()
        return likelihood, Likelihood_optim
        
    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

#         hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
