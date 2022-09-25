from datetime import datetime
from glob import glob
import time
import os
import torch
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
# from evaluate_model import compute_challenge_score
from torch import nn as nn
from torch.nn.modules.loss import _WeightedLoss
import matplotlib.pyplot as plt
import torch.nn.functional as F
plt.style.use('seaborn')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CutMixCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float() #.to(self.device)
        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        if len(target.shape)>1:
            target = target.argmax(dim=1)

        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss


class LDAMLoss(torch.nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # index = torch.zeros_like(x, dtype=torch.int64)
        # index.scatter_(1, target.data.view(-1, 1), 1)
        # index_float = index.type(torch.cuda.FloatTensor)

        index_float = x
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        #print(index)
        #print(x_m)
        #print(x)

        output = torch.where(index_float == 1.0, x_m, x)
      #  print(output)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

class Fitter:
    def __init__(self, model, device, config, params=None):
        self.params = params
        self.config = config
        self.epoch = 0
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        
        self.best_auc_murmur = 0
        self.best_auc_outcome = 0

        self.device = device
        self.model = model
        self.best_thresholds_murmur = None

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=0.0005)  # , momentum=0.9, weight_decay=0.0005)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.gaussian_loss = nn.GaussianNLLLoss()

        #self.loss = torch.nn.CrossEntropyLoss(reduction="none") #, weight=torch.tensor(config.class_weights).to(config.device))
        #self.loss = torch.nn.CrossEntropyLoss() # CrossEntropyLoss() #CutMixCrossEntropyLoss()
        #self.ldam_loss = LDAMLoss([149, 57, 579]) #CutMixCrossEntropyLoss()  #nn.L1Loss() #
        self.loss = CutMixCrossEntropyLoss()
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        #self.loss = FocalLoss(2, reduction="mean")
        #self.loss = LabelSmoothingLoss(classes=3, smoothing=0.1)
        #self.isClassification = isClassification
        #self.MAELoss = nn.L1Loss()

        self.log(f'Fitter prepared. Device is {self.device}. Optimizer is {self.optimizer}.')

    def fit(self, train_loader, validation_loader, model_nr, fold, innerfold):
        _tr = []
        _val = []
        _auc_murmur = []
        _auc_outcome = []
        _val_loss_murmur = []
        _murmur_class_1_auc = []
        _murmur_class_2_auc = []
        _murmur_class_3_auc = []
        patience = 0
        for e in range(self.config.n_epochs):

            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}, LR: {lr}')
            t = time.time()

            summary_loss = self.train_one_epoch(train_loader)
            _tr.append(summary_loss.avg)
            print(
                f'[RESULT]: Train. Epoch: {self.epoch}, loss: {summary_loss.avg:.3f}, time: {(time.time() - t):.1f}')

            t = time.time()
            summary_losses, auc_murmur, auc_outcome, classwise_murmur_aucs = self.validation(validation_loader)
            _murmur_class_1_auc.append(classwise_murmur_aucs[0])
            _murmur_class_2_auc.append(classwise_murmur_aucs[1])
            _murmur_class_3_auc.append(classwise_murmur_aucs[2])


            summary_loss = summary_losses[0].avg  # combined
            summary_loss1 = summary_losses[1].avg  # murmur
            summary_loss2 = summary_losses[2].avg  # outcome

            _val.append(summary_loss)
            _auc_murmur.append(auc_murmur)
            _auc_outcome.append(auc_outcome)
            print(
                f'[RESULT]: Val. Epoch: {self.epoch}, loss: {summary_loss:.3f}, m: {summary_loss1:.3f}, o: {summary_loss2:.3f}, auc_m {auc_murmur:.3f}, auc_o {auc_outcome:.3f}, score {round(0.00)} time: {(time.time() - t):.1f}')

            if auc_murmur > self.best_auc_murmur:
                self.best_auc_murmur = auc_murmur
                self.model.eval()
                self.save(f'{self.base_dir}/best-auc_murmur-model{model_nr}_fold{fold}_{innerfold}-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-auc_murmur-model{model_nr}_fold{fold}_{innerfold}-*epoch.bin'))[:-1]:
                    os.remove(path)

            """if auc_outcome > self.best_auc_outcome:
                self.best_auc_outcome = auc_outcome
                self.model.eval()
                self.save(f'{self.base_dir}/best-auc_outcome-fold{fold}_{innerfold}-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-auc_outcome-fold{fold}_{innerfold}-*epoch.bin'))[:-1]:
                    os.remove(path)"""

            if summary_loss < self.best_summary_loss:
                print("saving best model after best summmary loss 1")
                self.best_summary_loss = summary_loss
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-model{model_nr}_fold{fold}_{innerfold}-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-model{model_nr}_fold{fold}_{innerfold}-*epoch.bin'))[:-1]:
                    os.remove(path)

            """else:
                patience += 1
                print("patience:", patience)
                if patience > 10:
                    print("//////////////// Patience. Training done.")
                    break"""
                    
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss)

            # plot and save train log
            fig, ax = plt.subplots(ncols=1)
            ax.plot(np.arange(len(_tr)), np.array(_tr), label="train loss")
            ax.plot(np.arange(len(_tr)), np.array(_val), label="val loss")
            ax.plot(np.arange(len(_tr)), np.array(_auc_murmur), label="auc murmur", linestyle=":")
            ax.plot(np.arange(len(_tr)), np.array(_auc_outcome), label="auc outcome", linestyle=":")
            ax.plot(np.arange(len(_tr)), np.array(_murmur_class_1_auc), label="auc present", alpha=0.5)
            ax.plot(np.arange(len(_tr)), np.array(_murmur_class_2_auc), label="auc unknown", alpha=0.5)
            ax.plot(np.arange(len(_tr)), np.array(_murmur_class_3_auc), label="auc absent", alpha=0.5)
            plt.legend()
            #plt.grid()
            plt.ylim([0, 1])
            plt.xlim([0, self.config.n_epochs])
            plt.savefig(f'{self.base_dir}/loss_model{model_nr}_fold{fold}_{innerfold}.png', dpi=144)
            plt.close(fig)

            np.save(self.base_dir+f"/log_model{model_nr}_fold{fold}_{innerfold}.npy", np.array([_tr, _val, _auc_murmur, _auc_outcome]))
            self.epoch += 1
        return self.epoch

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_loss1 = AverageMeter()  # murmur
        summary_loss2 = AverageMeter()  # outcome
        t = time.time()

        gts_murmur = []
        pes_murmur = []

        gts_outcome = []
        pes_outcome = []

        # gt_shuffled = []
        for step, (images, input2, target) in enumerate(val_loader):
            #images = [[x.to(self.device) for x in bb] for bb in images]
            images = images.to(self.device)
            input2 = input2.to(self.device) #, dtype=torch.float)

            print(f'Val Step {step}/{len(val_loader)}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}', end="\r"
            )

            with torch.no_grad():
                batch_size = len(images)

                out = self.model(images, input2)

                y_murmur = target["murmur"].to(self.device).float()
                y_outcome = target["outcome"].to(self.device).float()
                y_where_hearable = target["wo_hörbar"].to(self.device).float()

                y_murmur_timing = target["aux_stuff"][0].to(self.device).float()
                y_murmur_shape = target["aux_stuff"][1].to(self.device).float()
                y_murmur_grading = target["aux_stuff"][2].to(self.device).float()
                y_murmur_pitch = target["aux_stuff"][3].to(self.device).float()
                y_murmur_quality = target["aux_stuff"][4].to(self.device).float()

                pred_murmur = out[0]
                pred_outcome = out[1]
                pred_where_hearable = out[2]
                pred_murmur_timing = out[3]
                pred_murmur_shape = out[4]
                pred_murmur_grading = out[5]
                pred_murmur_pitch = out[6]
                pred_murmur_quality = out[7]

                loss_murmur = self.loss(pred_murmur, y_murmur)
                loss_outcome = self.loss(pred_outcome, y_outcome)
                loss_where_hearable = self.bceloss(pred_where_hearable, y_where_hearable)
                loss_murmur_timing = self.loss(pred_murmur_timing, y_murmur_timing)
                loss_murmur_shape = self.loss(pred_murmur_shape, y_murmur_shape)
                loss_murmur_grading = self.loss(pred_murmur_grading, y_murmur_grading)
                loss_murmur_pitch = self.loss(pred_murmur_pitch, y_murmur_pitch)
                loss_murmur_quality = self.loss(pred_murmur_quality, y_murmur_quality)

                murmur_losses = (loss_murmur_timing + loss_murmur_shape + loss_murmur_grading + loss_murmur_pitch + loss_murmur_quality) / 5
                loss = (loss_murmur*3 + loss_outcome + loss_where_hearable + murmur_losses * 2) / 6
                if self.params=="only murmur loss":
                    loss = loss_murmur

                pes_murmur.extend(pred_murmur.softmax(dim=1).detach().cpu().numpy())
                gts_murmur.extend(y_murmur.detach().cpu().numpy())

                pes_outcome.extend(pred_outcome.softmax(dim=1).detach().cpu().numpy())
                gts_outcome.extend(y_outcome.detach().cpu().numpy())
                summary_loss.update(loss.detach().item(), batch_size)
                summary_loss1.update(loss_murmur.detach().item(), batch_size)
                summary_loss2.update(loss_outcome.detach().item(), batch_size)

        murmur_binary = False

        if murmur_binary:
            pes_murmur = np.array(pes_murmur)
            if len(np.array(gts_murmur).shape) > 1:

                gts_murmur = np.array(gts_murmur).argmax(axis=1)
                #print(np.array(gts_murmur).shape, "SHAPE1", pes_murmur.shape)
            else:
                #print(np.array(gts_murmur).shape, "SHAPE2", pes_murmur.shape)
                gts_murmur = np.array(gts_murmur)
            auc_murmur = roc_auc_score(gts_murmur, pes_murmur[:,1])
        else:
            pes_murmur = np.array(pes_murmur)
            if len(np.array(gts_murmur).shape) > 1:
                gts_murmur = np.array(gts_murmur).argmax(axis=1)
            else:
                gts_murmur = np.array(gts_murmur)
            auc_murmur = roc_auc_score(gts_murmur, pes_murmur, multi_class='ovr')

           # classwise_murmur_aucs = roc_auc_score(gts_murmur, pes_murmur, multi_class='ovr', average=None)
            classwise_murmur_aucs = np.array([0,0,0])


        pes_outcome = np.array(pes_outcome)[:, 1]
        if len(np.array(gts_outcome).shape) > 1:
            gts_outcome = np.array(gts_outcome).argmax(axis=1)
        else:
            gts_outcome = np.array(gts_outcome)
        auc_outcome = roc_auc_score(gts_outcome, pes_outcome)#, multi_class='ovr')
        """# competition metric
        gt_oh = []
        for i in gt:
            a = [0,0,0]
            a[i] = 1
            gt_oh.append(a)
        gt_oh = (np.array(gt_oh) != 0)
        pe_oh = []
        for i in pe:
            a = [0,0,0]
            a[np.argmax(i)] = 1
            pe_oh.append(a)        
        pe_oh = (np.array(pe_oh) != 0)
        classes = ['Present', 'Unknown', 'Absent']
        challenge_score = compute_challenge_score(gt_oh, pe_oh, classes)"""

        return [summary_loss, summary_loss1, summary_loss2], auc_murmur, auc_outcome, classwise_murmur_aucs #, challenge_score

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        #gt = []
        #pe = []
        for step, (images, input2, target) in enumerate(train_loader):
            images = images.to(self.device) #[[x.to(self.device) for x in bb] for bb in images]
            #images = [[x.to(self.device) for x in bb] for bb in images]
            input2 = input2.to(self.device) #, dtype=torch.float)

            print(f'Train Step {step}/{len(train_loader)}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}', end="\r"
            )

            batch_size = len(images)

            self.optimizer.zero_grad()
            out = self.model(images, input2)

            y_murmur = target["murmur"].to(self.device).float()
            y_outcome = target["outcome"].to(self.device).float()
            y_where_hearable = target["wo_hörbar"].to(self.device).float()

            y_murmur_timing = target["aux_stuff"][0].to(self.device).float()
            y_murmur_shape = target["aux_stuff"][1].to(self.device).float()
            y_murmur_grading = target["aux_stuff"][2].to(self.device).float()
            y_murmur_pitch = target["aux_stuff"][3].to(self.device).float()
            y_murmur_quality = target["aux_stuff"][4].to(self.device).float()

            pred_murmur = out[0]
            pred_outcome = out[1]
            pred_where_hearable = out[2]
            pred_murmur_timing = out[3]
            pred_murmur_shape = out[4]
            pred_murmur_grading = out[5]
            pred_murmur_pitch = out[6]
            pred_murmur_quality = out[7]

            loss_murmur = self.loss(pred_murmur, y_murmur)
            loss_outcome = self.loss(pred_outcome, y_outcome)
            loss_where_hearable = self.bceloss(pred_where_hearable, y_where_hearable)
            loss_murmur_timing = self.loss(pred_murmur_timing, y_murmur_timing)
            loss_murmur_shape = self.loss(pred_murmur_shape, y_murmur_shape)
            loss_murmur_grading = self.loss(pred_murmur_grading, y_murmur_grading)
            loss_murmur_pitch = self.loss(pred_murmur_pitch, y_murmur_pitch)
            loss_murmur_quality = self.loss(pred_murmur_quality, y_murmur_quality)

            murmur_losses = (loss_murmur_timing + loss_murmur_shape + loss_murmur_grading + loss_murmur_pitch + loss_murmur_quality) / 5
            loss = (loss_murmur*3 + loss_outcome + loss_where_hearable + murmur_losses * 2) / 6
            if self.params == "only murmur loss":
                loss = loss_murmur

            loss.backward()
            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()

            """# competition metric
            gt = target.detach().cpu().numpy()
            pe = output.detach().cpu().numpy()

            gt_oh = []
            for i in gt:
                a = [0,0,0]
                a[i] = 1
                gt_oh.append(a)
            gt_oh = (np.array(gt_oh) != 0)
            pe_oh = []
            for i in pe:
                a = [0,0,0]
                a[np.argmax(i)] = 1
                pe_oh.append(a)        
            pe_oh = (np.array(pe_oh) != 0)
            classes = ['Present', 'Unknown', 'Absent']
            challenge_score = compute_challenge_score(gt_oh, pe_oh, classes)
            #print("SCOOOORE;", challenge_score)"""

            if self.config.step_scheduler:
                self.scheduler.step()
        # print("")
        return summary_loss

    def make_prediction(self, crops):
        self.model.eval()
        crops = crops.to(self.device)
        masks = []
        with torch.no_grad():
            masks.append(self.model(crops))
        return masks

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': 0, # self.epoch  # when continue training, always start with epoch 0
            'best_thresh': self.best_thresholds_murmur,
        }, path)

    def log(self, message):
        if self.config.verbose:
            #sys.stdout.write(message)
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
            
    def load(self, path, silent = False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch']
        self.best_thresholds_murmur = checkpoint['best_thresh']
        if not silent:
            print("checkpoint loaded for epoch:", self.epoch - 1, "from path", path)