""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.augment_cnn import AugmentCNN
import copy


config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


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
        
        
        dataIndex = len(trn_y)+step*batch_size
        ignore_crit = nn.CrossEntropyLoss(reduction='none').cuda()       
        # forward
        logits,_ = self.net(trn_X)
        
        # sigmoid loss
        loss = torch.dot(torch.sigmoid(Likelihood[step*batch_size:dataIndex]), ignore_crit(logits, trn_y))/(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        
        loss.backward()
        dtloss_ll = Likelihood.grad
        
        
        dtloss_w = []
        # do virtual step (update gradient)       
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw in zip(self.net.weights(), self.v_net.weights()):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum  
                if w.grad is not None:
                    vw.copy_(w - xi * (m + w.grad )) 
                    dtloss_w.append(m + w.grad )
                elif w.grad is None:
                    dtloss_w.append(w.grad )
                
                
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
        
        
        logits, aux_logits = self.v_net(val_X)
        # calc unrolled loss
        ignore_crit = nn.CrossEntropyLoss(reduction='none').to(device)
        dataIndex = len(trn_y)+step*batch_size
        loss = torch.dot(torch.sigmoid(likelihood[step*batch_size:dataIndex]), ignore_crit(logits, trn_y))
        loss = loss/(torch.sigmoid(likelihood[step*batch_size:dataIndex]).sum()) # L_val(w`)
               
        # compute gradient
        loss.backward()
                
        dvloss_tloss = 0  
        for v, dt in zip(self.v_net.weights(), dtloss_w):
            if v.grad is not None:
                grad_valw_d_trainw = torch.div(v.grad, dt)
                grad_valw_d_trainw[torch.isinf(grad_valw_d_trainw)] = 0
                grad_valw_d_trainw[torch.isnan(grad_valw_d_trainw)] = 0
                grad_val_train = torch.sum(grad_valw_d_trainw)
#                 print(grad_val_train)
                dvloss_tloss += grad_val_train
            
            
        dlikelihood = dvloss_tloss* dtloss_ll
        
        vprec1, vprec5 = utils.accuracy(logits, val_y, topk=(1, 5))
        
        Likelihood_optim.zero_grad()
        likelihood.grad = dlikelihood
        print(dvloss_tloss)
        print(dtloss_ll)
        print('likelihood gradient is:', likelihood.grad)
        Likelihood_optim.step()
        return likelihood, Likelihood_optim, loss, vprec1, vprec5
        
    
def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_val_data, test_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True)

    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype).to(device)       #single GPU
#     model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer with SGD
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    
    n_train = len(train_val_data)
    split = n_train // 2
    indices = list(range(n_train))
    
    # each train data is endowed with a weight
    Likelihood = torch.nn.Parameter(torch.ones(len(indices[:split])).cuda(),requires_grad=True)
    Likelihood_optim = torch.optim.SGD({Likelihood}, config.lr)
   
    # data split
    train_data = torch.utils.data.Subset(train_val_data, indices[:split])
    valid_data = torch.utils.data.Subset(train_val_data, indices[split:])
    
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=False)
        
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    architect = Architect(model, 0.9, 3e-4)
    
    best_top1 = 0.
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.drop_path_prob(drop_prob)

        # training
        train(train_loader, valid_loader, model, architect, optimizer, criterion, lr, epoch, Likelihood, Likelihood_optim, config.batch_size)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, valid_loader, model, architect, optimizer, criterion, lr, epoch, Likelihood, Likelihood_optim, batch_size):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    standard_losses = utils.AverageMeter()
    valid_losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()
    
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)
        M = val_X.size(0)

        # phase 2. Likelihood step (Likelihood)
        Likelihood_optim.zero_grad()
        Likelihood, Likelihood_optim, valid_loss, vprec1, vprec5= architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, optimizer, model, Likelihood, Likelihood_optim, batch_size, step)
            
            
        # phase 1. network weight step (w)    
        optimizer.zero_grad()
        logits, aux_logits = model(trn_X)      
        
        ignore_crit = nn.CrossEntropyLoss(reduction='none').to(device)
        dataIndex = len(trn_y)+step*batch_size
        loss = torch.dot(torch.sigmoid(Likelihood[step*batch_size:dataIndex]), ignore_crit(logits, trn_y))
        loss = loss/(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        '''
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        '''
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # update network weight on train data
        optimizer.step()     
        
        #compare normal loss without weighted
        standard_loss = criterion(logits, trn_y)
            
        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        standard_losses.update(standard_loss.item(), N)
        valid_losses.update(valid_loss.item(), M)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
    
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} standard Loss {slosses.avg:.3f} Valid Loss {vlosses.avg:.3f}"
                " Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses, slosses=standard_losses, vlosses=valid_losses,
                    top1=top1, top5=top5))
        

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        writer.add_scalar('val/loss', valid_loss.item(), cur_step)
        writer.add_scalar('train/top1', vprec1.item(), cur_step)
        writer.add_scalar('train/top5', vprec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step,(X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0) 

            logits, _ = model(X)
            loss = criterion(logits, y)
            
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Test: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('test/loss', losses.avg, cur_step)
    writer.add_scalar('test/top1', top1.avg, cur_step)
    writer.add_scalar('test/top5', top5.avg, cur_step)

    logger.info("Test: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
