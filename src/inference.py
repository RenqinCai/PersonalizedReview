import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu

class INFER(object):
    def __init__(self, sos_idx, eos_idx, pad_idx, i2w, eval_data, args, device):
        super().__init__()

        self.m_sos_idx = sos_idx
        self.m_eos_idx = eos_idx
        self.m_pad_idx = pad_idx
        self.m_i2w = i2w

        self.m_eval_data = eval_data
        self.m_epoch = args.epochs
        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_anneal_func = args.anneal_func
        self.m_device = device

    def f_init_infer(self, network, reload_model=False):
        if reload_model:
            print("reload model")
        self.m_network = network

        # self.m_Recon_loss_fn = Reconstruction_loss(self.m_device)
        # self.m_KL_loss_fn = KL_loss(self.m_device)
        # self.m_RRe_loss_fn = RRe_loss(self.m_device)
        # self.m_ARe_loss_fn = ARe_loss(self.m_device)

    def f_inference(self):
        self.m_mean_loss = 0
        for epoch_i in range(self.m_epoch):
            loss_epoch = self.f_inference_epoch()
            
    def f_inference_epoch(self):
        # batch_size = args.batch_size

        infer_loss_list = []

        for input_batch, target_batch, ARe_batch, RRe_batch, length_batch in self.m_eval_data:
            input_batch = input_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
            RRe_batch = RRe_batch.to(self.m_device)
            ARe_batch = ARe_batch.to(self .m_device)

            logp, z_mean, z_logv, z, s_mean, s_logv, s, ARe_pred, RRe_pred = self.m_network(input_batch, length_batch)
            print(" "*10, "*"*10, " "*10)
            
            print("->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

            # mean = mean.unsqueeze(0)
            mean = torch.cat([z_mean, s_mean], dim=1)
            max_seq_len = max(length_batch)
            samples, z = self.f_decode_text(mean, max_seq_len)

            # print("->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

            print("<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

            
    def f_decode_text(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size

        hidden = self.m_network.m_latent2hidden(z)
        
        hidden = hidden.unsqueeze(0)

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0
        init_hidden = hidden

        # print("hidden size", hidden.size())

        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_embedding = self.m_network.m_embedding(input_seq)

            # print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_decoder_rnn(input_embedding, hidden)

            logits = self.m_network.m_output2vocab(output)

            input_seq = self._sample(logits)

            if len(input_seq.size()) < 1:
                input_seq = input_seq.unsqueeze(0)

            generations = self._save_sample(generations, input_seq, seq_running, t)

            seq_mask[seq_running] = (input_seq != self.m_eos_idx).bool()
            seq_running = seq_idx.masked_select(seq_mask)

            running_mask = (input_seq != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_seq = input_seq[running_seqs]

                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)

            t += 1

        return generations, z

    def _sample(self, dist, mode="greedy"):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):
        # print(" "*10, "*"*10)
        for word_id in sent:

            if word_id == pad_idx:
                break
            # print('word_id', word_id.item())
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str