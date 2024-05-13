# -*- coding: utf-8 -*-

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
import numpy as np
import soundfile as sf
from utils.tools import *
from utils.CNN import CNN
from utils.RawNetModel import RawNet3
import pandas as pd
from utils.loss import AAMsoftmax
import torch
import torch.nn as nn
from utils.ECAPA import ECAPA_TDNN
from asteroid_filterbanks import Encoder, ParamSincFB
from utils.RawNetBasicBlock import Bottle2neck, PreEmphasis
class FakeModel(nn.Module):
    def __init__(self, lr, lr_decay, n_class, device, test_step, **kwargs):
        super(FakeModel, self).__init__()
        ## ResNet
        self.device = device
        
        self.speaker_encoder = RawNet3(Bottle2neck,model_scale=8,context=True,summed=True,encoder_type="ECA",nOut=256,out_bn=False,sinc_stride=10,log_sinc=True,norm_sinc="mean",grad_mult=1,)
        # self.speaker_encoder = ECAPA_TDNN(C=1024).to(self.device)
        self.speaker_loss = AAMsoftmax(n_class=2, m=0.2, s=30).to(self.device)
        self.class_loss = nn.CrossEntropyLoss()
        # ## Classifier
        # self.speaker_loss = nn.CrossEntropyLoss()
        
        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, acc, loss, recall, F1 = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start = 1):
            self.zero_grad()
            labels            = torch.LongTensor(labels).to(self.device)
            
            speaker_emb = self.speaker_encoder.forward(data.to(self.device))
            outputs = self.speaker_loss(speaker_emb)
            nloss = self.class_loss(outputs, labels)
            acc_t, recall_t, prec_t, F1_t = metrics_scores(outputs, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            acc += acc_t
            recall += recall_t
            F1 += F1_t
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, Acc: %2.2f%%, Recall: %2.2f%%, F1: %2.2f%%\r" %(loss/(num), acc/index*len(labels), recall/index*len(labels), F1/index*len(labels)*100))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr, acc/index*len(labels), recall/index*len(labels), F1/index*len(labels)*100 

    def eval_network(self, eval_list, **kwargs):
        
        model = RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1,
        )
        files = []
        outputs = torch.tensor([]).to(self.device)
        df_test = pd.read_csv(eval_list)
        label_list = df_test["label"].tolist()
        setfiles = df_test["wav_path"].tolist()
        loss, top1 = 0, 0
        model.load_state_dict(
            torch.load(
                "/home/czy/data/contests/deepfake/model/2024_finvcup_baseline/pretrain/model.pt",
                map_location=lambda storage, loc: storage,
            )["model"]
        )
        model.eval()
        print("RawNet3 initialised & weights loaded!")
        if torch.cuda.is_available():
            print("Cuda available, conducting inference on GPU")
            model = model.to("cuda")
            gpu = True
            # 提取说话人embedding
        
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _ = soundfile.read(file)
            output = self.extract_speaker_embd(model=model, fn=file, gpu=gpu).mean(0)
            print(f"file: {file}, output.shape: {output.shape}") # [1, 256]
            output = self.speaker_loss(output) # [5, 2]
            output = torch.mean(output, dim=0).view(1, -1) # [1, 2]
        
            outputs = torch.cat((outputs, output), 0)
        
        print(f"outputs.shape: {outputs.shape}")
        acc, recall, prec, F1 = metrics_scores(outputs, torch.tensor(label_list).to(self.device))

        return acc, recall, F1*100

    
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
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
    
    
    def extract_speaker_embd(
        self, model, fn, n_samples: int = 48000, n_segments: int = 10, gpu: bool = False
    ) -> np.ndarray:
        print(fn)
        audio, sample_rate = sf.read(fn)
        # 检查是否单声道
        if len(audio.shape) > 1:
            raise ValueError(
                f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
            )
        # 检查音频采样率是否是16k
        if sample_rate != 16000:
            raise ValueError(
                f"RawNet3 supports 16k sampling rate only. Input data's sampling rate is {sample_rate}."
            )
        # 音频长度不足时进行填充
        if (
            len(audio) < n_samples
        ):  # RawNet3 was trained using utterances of 3 seconds
            shortage = n_samples - len(audio) + 1
            audio = np.pad(audio, (0, shortage), "wrap")

        audios = []
        startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
        for asf in startframe:
            audios.append(audio[int(asf) : int(asf) + n_samples])

        audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
        if gpu:
            audios = audios.to("cuda")
        with torch.no_grad():
            output = model(audios)
            print(output.shape)
        return output