import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from W_Construct import KNN,norm_W
from data_loader import load_mat
import numpy as np
from sklearn import metrics
import warnings
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='MVC')
parser.add_argument('--epochs', '-te', type=int, default=30, help='number of train_epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
parser.add_argument('--r', type=float, default=-1, help='Scalar to control the distribution of the weights, setting 1 on Yale')
args = parser.parse_args()

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=64):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        context = torch.bmm(attention, v)

        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=64, num_heads=8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)
        self._initialize_weights()

    def forward(self, key, value, query):

        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(num_heads, -1, dim_per_head)
        value = value.view( num_heads, -1, dim_per_head)
        query = query.view(num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale)

        # concat heads
        context = context.view(-1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=64, ffn_dim=512, dropout=0.2):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self._initialize_weights()

    def forward(self, x):
        output = x
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):

        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class Encoder(nn.Module):
    def __init__(self,
               num_size,
               num_layers=1,
               model_dim=64,
               num_heads=8,
               ffn_dim=512,
               dropout=0.2):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Linear(num_size, model_dim)

    def forward(self, inputs):
        output = self.seq_embedding(inputs)

        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)

        return output, attentions

class GRTrans(nn.Module):
    def __init__(self,n_sample,n_cluster,lam1):
        super(GRTrans, self).__init__()

        self.encoder = Encoder(num_size=n_sample)

        self.l1 = nn.Linear(64,n_cluster)

        self.weight = nn.Parameter(torch.full((n_view,), 1.0), requires_grad=True)

        self.lam1 = lam1


    def forward(self,W):

        weight = F.softmax(self.weight)

        weight = torch.pow(weight,-1)

        W_new = torch.matmul(W,weight)

        CF,_= self.encoder(W_new)

        CF = F.softmax(self.l1(CF), dim=1)

        return CF,W_new

    def run(self,W):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        obj =[]
        self.to(device)
        for it in range(args.epochs):
            for i in range(100):
                optimizer.zero_grad()
                CF,W_new = self(W)
                E = CF@CF.t()
                loss_gr = torch.pow(torch.norm((E - W_new)),2)
                loss_or = 2 / (n_cluster * (n_cluster - 1)) * triu(torch.t(CF) @ CF)
                loss = loss_gr+loss_or*self.lam1
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            y_pred = np.argmax(CF.cpu().detach().numpy(), axis=1) + 1
            ACC = acc(GT, y_pred)
            NMI = metrics.normalized_mutual_info_score(GT, y_pred)
            Purity = purity_score(GT, y_pred)
            ARI = metrics.adjusted_rand_score(GT, y_pred + 1)
            obj.append(loss.cpu().detach().numpy())
            print('epoch: {},clustering accuracy: {}, NMI: {}, Purity: {}, ARI: {}'.format(it,ACC, NMI, Purity,ARI))
        return CF,ACC,NMI,Purity,ARI,y_pred,W_new,obj,E



def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    return metrics.accuracy_score(y_true, y_voted_labels)

def triu(X):
    return torch.sum(torch.triu(X, diagonal=1))

if __name__ =="__main__":
    X, GT = load_mat('data/flower17.mat')
    n_cluster =17
    N = X[0].shape[0]
    GT = GT.reshape(np.max(GT.shape), )
    n_view = len(X)
    W = torch.zeros((N, N, n_view))
    ACC = []
    NMI = []
    P = []
    ARI = []
    for i in range(n_view):
        A = KNN(X[i],20)
        W[:, :, i] = torch.tensor(norm_W(A), dtype=torch.float32)
    lambda_1 = [0,0.1,1,10,100,1000]
    for j in range(6):
        for t in range(10):
            model = GRTrans(N,n_cluster,lambda_1[j])
            CF,a,n,p,ari,y_pred,W_new,obj,E= model.run(W.to(device))
            ACC.append(a)
            NMI.append(n)
            P.append(p)
            ARI.append(ari)
        print(np.array(ACC).mean(),np.array(ACC).std())
        print(np.array(NMI).mean(), np.array(NMI).std())
        print(np.array(P).mean(), np.array(P).std())
        print(np.array(ARI).mean(), np.array(ARI).std())
