import os
import json
import time
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from metric import Reconstruction_loss, KL_loss, RRe_loss, ARe_loss
from model import REVIEWDI
from inference import INFER

class TRAINER(object):

    def __init__(self, train_data, eval_data, network, pad_id, args, device):
        super().__init__()

        self.m_train_data = train_data
        self.m_eval_data = eval_data
        self.m_network = network

        self.m_pad_index = pad_id

        self.m_data = {"train": self.m_train_data, "val":self.m_eval_data}
        self.m_epochs = 0

        self.m_optim = None
        self.m_Recon_loss_fn = None
        self.m_KL_loss_fn = None
        self.m_RRe_loss_fn = None
        self.m_ARe_loss_fn = None
        self.m_infer = None

        self.m_save_mode = True
        self.m_mean_train_loss = 0
        self.m_mean_val_loss = 0
        
        self.m_device = device

        self.m_epochs = args.epochs
        self.m_batch_size = args.batch_size

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_anneal_func = args.anneal_func

        if args.optimizer == "Adam":
            self.m_optim = torch.optim.Adam(self.m_network.parameters(), lr=args.learning_rate)

        self.m_Recon_loss_fn = Reconstruction_loss(self.m_device, ignore_index=self.m_pad_index)
        self.m_KL_loss_fn = KL_loss(self.m_device)
        self.m_RRe_loss_fn = RRe_loss(self.m_device)
        self.m_ARe_loss_fn = ARe_loss(self.m_device)

        self.m_step = 0

    def f_train(self):

        last_train_loss = 0
        last_eval_loss = 0

        for epoch in range(self.m_epochs):
            print("+"*20)

            self.f_train_epoch("train")
            
            if last_train_loss == 0:
                last_train_loss = self.m_mean_train_loss

            elif last_train_loss < self.m_mean_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                last_train_loss = self.m_mean_train_loss

            # self.m_infer.f_inference_epoch()
            self.f_train_epoch("val")

            if last_eval_loss == 0:
                last_eval_loss = self.m_mean_val_loss
            elif last_eval_loss < self.m_mean_val_loss:
                print("!"*10, "overfitting validation loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
            else:
                
                print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
                last_eval_loss = self.m_mean_val_loss

    def f_train_epoch(self, train_val_flag):

        train_loss_list = []
        eval_loss_list = []

        batch_size = self.m_batch_size

        for input_batch, target_batch, ARe_batch, RRe_batch, length_batch in self.m_data[train_val_flag]:
            # print("training")
            # exit()
            if train_val_flag == "train":
                self.m_network.train()
            elif train_val_flag == "val":
                self.m_network.eval()
            else:
                raise NotImplementedError

            self.m_step += 1
            
            input_batch = input_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
            RRe_batch = RRe_batch.to(self.m_device)
            ARe_batch = ARe_batch.to(self.m_device)

            logp, z_mean, z_logv, z, s_mean, s_logv, s, ARe_pred, RRe_pred = self.m_network(input_batch, length_batch)

            ### NLL loss
            NLL_loss = self.m_Recon_loss_fn(logp, target_batch, length_batch)

            ### KL loss
            KL_loss, KL_weight = self.m_KL_loss_fn(s_mean, s_logv, self.m_step, self.m_k, self.m_x0, self.m_anneal_func)

            ### RRe loss
            RRe_loss = self.m_RRe_loss_fn(RRe_pred, RRe_batch)

            ### ARe loss
            ARe_loss = self.m_ARe_loss_fn(ARe_pred, ARe_batch)

            loss = (NLL_loss+KL_loss*KL_weight+RRe_loss+ARe_loss)/batch_size
            
            if train_val_flag == "train":
                self.m_optim.zero_grad()
                loss.backward()
                self.m_optim.step()

                train_loss_list.append(loss.item())
                self.m_mean_train_loss = np.mean(train_loss_list)
            elif train_val_flag == "val":
                eval_loss_list.append(loss.item())
                self.m_mean_val_loss = np.mean(eval_loss_list)

            
        


