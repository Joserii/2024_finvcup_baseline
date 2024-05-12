'''
This part is used to train the speaker model and evaluate the performances
'''

import torch
import sys
import os
import time
import torch.nn as nn
import pandas as pd
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
from .loss import AAMsoftmax
from .tools import metrics_scores
from .ECAPA_TDNN import ECAPA_TDNN
import torchmetrics


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, device, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        # ECAPA-TDNN
        self.device = device
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        # Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()

        self.optim = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, acc, loss, recall, F1 = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(
                data.cuda(), aug=True)
            nloss, prec, output = self.speaker_loss.forward(
                speaker_embedding, labels)
            acc_t, recall_t, prec_t, F1_t = metrics_scores(output, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            acc += acc_t
            recall += recall_t
            F1 += F1_t
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) +
                             " Loss: %.5f, ACC: %2.2f%%  \r" % (loss/(num), acc/index*len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr, acc/index*len(labels), recall/index*len(labels), F1/index*len(labels)*100 

    def eval_network(self, loader):
        self.eval()
        device = torch.device('cuda:0')
        test_acc = torchmetrics.Accuracy(
            task="binary", num_classes=2).to(device)
        test_recall = torchmetrics.Recall(
            task="binary", average='none', num_classes=2).to(device)
        test_precision = torchmetrics.Precision(
            task="binary", average='none', num_classes=2).to(device)
        #test_auc = torchmetrics.AUROC(task="binary",average="macro", num_classes=2).to(device)
        size = len(loader.dataset)
        num_batches = len(loader)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for (data, labels) in loader:
                labels = torch.LongTensor(labels).cuda()
                speaker_embedding = self.speaker_encoder.forward(
                    data.cuda(), aug=True)
                nloss, prec, output = self.speaker_loss.forward(
                    speaker_embedding, labels)
                test_loss += nloss
                correct += (output.argmax(1) ==
                            labels).type(torch.float).sum().item()
                # 一个batch进行计算迭代
                test_acc(output.argmax(1), labels).to(device)
                # test_auc.update(output, labels)
                test_recall(output.argmax(1), labels).to(device)
                test_precision(output.argmax(1), labels).to(device)
        test_loss /= num_batches
        correct /= size
        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()
        F1 = 2*total_precision*total_recall/(total_recall+total_precision)
        # total_auc = test_auc.compute()
        print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, "
              f"Avg loss: {test_loss:>8f}, "
              f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
        print("recall of every test dataset class: ", total_recall)
        print("precision of every test dataset class: ", total_precision)
        print("F1:", F1)

        sys.stdout.write("\n")
        return total_precision, total_recall, F1

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
