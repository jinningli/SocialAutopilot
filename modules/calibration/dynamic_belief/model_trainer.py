import os
import numpy as np
import scipy.sparse as sp
import torch
import time
import json

from evaluate import Evaluator
from model import InfoVGAE, Discriminator
from PID import PIDControl

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

def sp_sparse_to_torch_longtensor(coo_matrix):
    i = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    v = torch.LongTensor(coo_matrix.data)
    return torch.sparse.LongTensor(i, v, torch.Size(coo_matrix.shape))

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

class TrainerBase():
    def __init__(self):
        self.name = "TrainerBase"

    def train(self):
        raise NotImplementedError(self.name)


class InfoVGAETrainer(TrainerBase):
    def __init__(self, adj_matrix, features, args, dataset):
        super(InfoVGAETrainer).__init__()
        self.name = "InfoVGAETrainer"
        self.adj_matrix = adj_matrix
        self.features = features
        self.args = args
        self.dataset = dataset  # for freeze usage

        self.model = None
        self.optimizer = None

        self.result_embedding = None
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=args.output_path + "/runs")
        # tensorboard --logdir=runs

    def train(self):
        print("Training using {}".format(self.name))

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = self.adj_matrix
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train = self.adj_matrix
        adj = adj_train

        # Some preprocessing
        adj_norm = preprocess_graph(adj)

        features = sparse_to_tuple(sp.coo_matrix(self.features))

        # Create Model
        pos_weight = self.args.pos_weight_lambda * float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        print("Positive sample weight: {}".format(pos_weight))

        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                             torch.FloatTensor(adj_label[1]),
                                             torch.Size(adj_label[2]))
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2]))

        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight
        ones = torch.ones(self.adj_matrix.shape[0], dtype=torch.long)
        zeros = torch.zeros(self.adj_matrix.shape[0], dtype=torch.long)

        if self.args.use_cuda:
            adj_norm = adj_norm.cuda()
            adj_label = adj_label.cuda()
            features = features.cuda()
            weight_tensor = weight_tensor.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()

        # init model and optimizer
        self.model = InfoVGAE(self.args, adj_norm)
        if self.args.use_cuda:
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        # train model
        # Kp = 0.001
        # Ki = -0.001
        # PID = PIDControl(Kp, Ki)
        # Exp_KL = 0.005

        last_loss = None

        for epoch in range(self.args.epochs):
            t = time.time()

            # Train VAE
            z = self.model.encode(features)

            if self.args.freeze_dict is not None:
                z_freezed = torch.zeros_like(z)
                z_freezed[~self.dataset.freeze_mask] = z[~self.dataset.freeze_mask]
                z_freezed[self.dataset.freeze_mask] = self.dataset.freeze_tensor
                A_pred = self.model.decode(z_freezed)
            else:
                A_pred = self.model.decode(z)

            vae_recon_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                           weight=weight_tensor)
            kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * self.model.logstd - self.model.mean ** 2 -
                                                    torch.exp(self.model.logstd) ** 2).sum(1).mean()
            # weight = PID.pid(Exp_KL, kl_divergence.item())  # get the weight on KL term with PI module
            vae_loss = vae_recon_loss - kl_divergence

            if last_loss is not None and np.abs(vae_loss.item() - last_loss) < 1e-6:
                print("Early stop {} -> {}".format(vae_loss.item(), last_loss))
                break

            self.optimizer.zero_grad()
            vae_loss.backward()
            self.optimizer.step()

            # if epoch % 20 == 0:
                # self.writer.add_embedding(self.model.encode(features).detach().cpu().numpy(), global_step=epoch)

            if epoch % 1 == 0:
                log = "Epoch: {}, loss_recon: {:.5f}, loss_kl: {:.5f}, loss_VAE: {:.5f}".format(
                        epoch,
                        vae_recon_loss.item(),
                        - kl_divergence,
                        vae_loss.item())
                print(log)
                with open(self.args.output_path + "/log.txt", "a") as fout:
                    fout.write("Epoch: {}\n".format(epoch))
                    fout.write(log + "\n\n")

            last_loss = vae_loss.item()

        self.writer.close()
        self.result_embedding = self.model.encode(features).detach().cpu().numpy()
        if self.args.freeze_dict is not None:
            # print(self.result_embedding[self.dataset.freeze_mask])
            # print(self.dataset.freeze_tensor)
            self.result_embedding[self.dataset.freeze_mask] = self.dataset.freeze_tensor.numpy()


    def save(self, path=None):
        path = self.args.output_path if path is None else path
        # Save result embedding of nodes
        with open(path + "/args.json", 'w') as fout:
            json.dump(vars(self.args), fout)
        with open(path + "/embedding.bin", 'wb') as fout:
            pickle.dump(self.result_embedding, fout)
            print("Embedding and dependencies are saved in {}".format(path))
        with open(path + "/freeze_dict.pkl", 'wb') as fout:
            output_dict = {}
            for i, tweet_id in enumerate(self.dataset.tweet_id_list):
                output_dict[tweet_id] = self.result_embedding[self.dataset.num_user + i]
            pickle.dump(output_dict, fout)
            print("freeze_dict saved in {}".format(path))

    def get_scores(self, adj_orig, edges_pos, edges_neg, adj_rec):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(self, adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy
