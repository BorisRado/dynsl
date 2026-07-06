import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, feats, mlp_hidden, head=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats, head=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class InitialLayer(nn.Module):
    def __init__(self, f, hidden, num_tokens, patch_size, patch):
        super().__init__()
        self.patch_size = patch_size
        self.patch = patch
        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        return out

    def _to_words(self, x):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out


class TokenExtractor(nn.Module):
    def forward(self, x):
        return x[:,0]


class ViT(nn.Sequential):
    """Adaptation of https://github.com/omihub777/ViT-CIFAR"""
    def __init__(self, num_classes=10, patch=8, dropout:float=0.1, num_layers=7, hidden=384, mlp_hidden=384, head=12, use_groupnorm=False):
        super().__init__()

        img_size = 32
        f = (img_size // patch) ** 2 * 3 # 48 # patch vec length
        num_tokens = (patch**2) + 1


        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )
        super().__init__(
            InitialLayer(f, hidden, num_tokens, patch_size=img_size//patch, patch=patch),
            *enc_list,
            TokenExtractor(),
            fc
        )
        self.meaningful_split_points = list(range(2, num_layers+1)) + [num_layers+3,]
        self._data_type = "cv"
