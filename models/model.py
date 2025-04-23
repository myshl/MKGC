import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig
from utils import get_performance, get_loss_fn, GRAPH_MODEL_CLASS
from models.soft_prompter import Prompter
from models.hierarchy_model import HierarchyModel
import numpy as np
import pickle
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class MKGCModel(pl.LightningModule):
    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']

        self.ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)
        path = 'img_data/'
        img_features = pickle.load(open(path + self.configs.dataset + '_img_features.pkl', 'rb'))
        img_features = F.normalize(torch.Tensor(img_features), p=2, dim=1)
        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(img_features.view(-1, 64, 64))
        img = img.view(img.size(0), -1)

        self.img_ent_embed = torch.nn.Embedding.from_pretrained(img, freeze=False)
        gat_entity = torch.tensor(pickle.load(open(path + self.configs.dataset + '_entity_embed.pkl', 'rb')))
        self.gat_relation = torch.tensor(pickle.load(open(path + self.configs.dataset + '_relation_embed.pkl', 'rb')))
        self.gat_ent_embed = nn.Embedding.from_pretrained(gat_entity, freeze=False)

        if self.configs.graph_model in ['transe', 'rotate']:
            self.img_rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
            self.gat_rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.img_rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)
            self.gat_rel_embed = nn.Embedding.from_pretrained(self.gat_relation, freeze=False)

        if self.configs.graph_model in ['transe', 'rotate']:
            self.rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = self.configs.prompt_length
        self.plm_configs.prompt_hidden_dim = self.configs.prompt_hidden_dim
        self.plm = HierarchyModel.from_pretrained(configs.pretrained_model)

        self.prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.img_prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.gat_prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.fc = nn.Linear(configs.prompt_length * self.plm_configs.hidden_size, configs.embed_dim)

        if configs.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)

        self.history = {'perf': ..., 'loss': []}
        self.loss_fn = get_loss_fn(configs)
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9
        if self.configs.alpha_step > 0:
            self.alpha = 0.
        else:
            self.alpha = self.configs.alpha

        # self.all_embeddings = []  # To store embeddings from all batches
        # self.all_triples = []  # To store corresponding triples


    def forward(self, ent_rel, src_ids, src_mask):
        bs = ent_rel.size(0)
        all_ent_embed = self.gat_ent_embed.weight


        img_ent_embed = self.img_ent_embed.weight
        gat_ent_embed = self.gat_ent_embed.weight

        if self.configs.graph_model in ['transe', 'rotate']:
            all_rel_embed = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)
            img_rel_embed = torch.cat([self.img_rel_embed.weight, -self.img_rel_embed.weight], dim=0)
            gat_rel_embed = torch.cat([self.gat_rel_embed.weight, -self.gat_rel_embed.weight], dim=0)

        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            all_rel_embed = self.rel_embed.weight
            img_rel_embed = self.img_rel_embed.weight
            gat_rel_embed = self.gat_rel_embed.weight


        ent, rel = ent_rel[:, 0], ent_rel[:, 1]

        # Constructing multimodal prompts for hierarchical interaction
        img_ent_embed = img_ent_embed[ent]
        img_rel_embed = img_rel_embed[rel]
        img_prompt = self.img_prompter(torch.stack([img_ent_embed, img_rel_embed], dim=1))
        gat_ent_embed = gat_ent_embed[ent]
        gat_rel_embed = gat_rel_embed[rel]
        gat_prompt = self.gat_prompter(torch.stack([gat_ent_embed, gat_rel_embed], dim=1))
        prompt = torch.cat([img_prompt, gat_prompt], dim=1)
        prompt_attention_mask = torch.ones(img_ent_embed.size(0), self.configs.prompt_length * 2).type_as(src_mask)
        src_mask = torch.cat((prompt_attention_mask, src_mask), dim=1)
        output = self.plm(input_ids=src_ids, attention_mask=src_mask, layerwise_prompt=prompt)
        last_hidden_state = output.last_hidden_state

        ent_rel_state = last_hidden_state[:, :self.configs.prompt_length * 2]
        plm_ent_embed, plm_rel_embed = torch.chunk(ent_rel_state, chunks=2, dim=1)
        plm_ent_embed = self.fc(plm_ent_embed.reshape(img_ent_embed.size(0), -1))
        plm_rel_embed = self.fc(plm_rel_embed.reshape(img_rel_embed.size(0), -1))
        pred = self.graph_model(plm_ent_embed, plm_rel_embed)
        logits = self.graph_model.get_logits(pred, all_ent_embed)
        return logits

    def training_step(self, batched_data, batch_idx):
        if self.configs.alpha_step > 0 and self.alpha < self.configs.alpha:
            self.alpha = min(self.alpha + self.configs.alpha_step, self.configs.alpha)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        ent_rel = batched_data['ent_rel']
        tgt_ent = batched_data['tgt_ent']
        labels = batched_data['labels']
        logits, pred = self(ent_rel, src_ids, src_mask)
        loss = self.loss_fn(logits, labels)
        self.history['loss'].append(loss.detach().item())

        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        test_triples = batched_data['triple']
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        tgt_ent = batched_data['tgt_ent']
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        logits = self(ent_rel, src_ids, src_mask)
        logits = logits.detach()
        # if self.configs.visualize_embeddings:
        #     entity_embeddings = torch.cat(self.all_embeddings, dim=0).cpu().numpy()
        #     if self.configs.embedding_dim_reduction == 'PCA':
        #         pca = PCA(n_components=2)
        #         reduced_embeddings = pca.fit_transform(entity_embeddings)
        #     elif self.configs.embedding_dim_reduction == 'tSNE':
        #         tsne = TSNE(n_components=2, random_state=42)
        #         reduced_embeddings = tsne.fit_transform(entity_embeddings)
        #     else:
        #         raise ValueError("Invalid embedding_dim_reduction method. Choose 'PCA' or 'tSNE'.")
        #     self.save_embedding_plot(reduced_embeddings)
            
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                tgt_filter = gt[(hi, ri)]
            tgt_score = logits[i, ti].item()
            logits[i, tgt_filter] = self._MASKING_VALUE
            logits[i, ti] = tgt_score
        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            rank = np.where(argsort[i] == ti)[0][0] + 1
            ranks.append(rank)
        if self.configs.use_log_ranks:
            filename = os.path.join(self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp')
            self.log_ranks(filename, test_triples, argsort, ranks, batch_idx)
        return ranks

    def validation_epoch_end(self, outs):
        tail_ranks = np.concatenate(outs[0])
        head_ranks = np.concatenate(outs[1])
        perf = get_performance(self, tail_ranks, head_ranks)
        print('Epoch:', self.current_epoch)
        print(perf)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.configs.lr)

    def log_ranks(self, filename, test_triples, argsort, ranks, batch_idx):
        assert len(test_triples) == len(ranks), 'length mismatch: test_triple, ranks!'
        with open(filename, 'a') as file:
            for i, triple in enumerate(test_triples):
                if not self.configs.is_temporal:
                    head, tail, rel = triple
                    timestamp = ''
                else:
                    head, tail, rel, timestamp = triple
                    timestamp = ' | ' + timestamp
                rank = ranks[i].item()
                triple_str = self.ent_names[head] + ' [' + self.ent_descs[head] + '] | ' + self.rel_names[rel]\
                    + ' | ' + self.ent_names[tail] + ' [' + self.ent_descs[tail] + '] ' + timestamp + '(%d %d %d)' % (head, tail, rel)
                file.write(str(batch_idx * self.configs.val_batch_size + i) + '. ' + triple_str + '=> ranks: ' + str(rank) + '\n')

                best10 = argsort[i, :10]
                for ii, ent in enumerate(best10):
                    ent = ent.item()
                    mark = '*' if (ii + 1) == rank else ' '
                    file.write('\t%2d%s ' % (ii + 1, mark) + self.ent_names[ent] + ' [' + self.ent_descs[ent] + ']' + ' (%d)' % ent + '\n')

