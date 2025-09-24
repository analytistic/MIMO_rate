import torch
import torch.nn as nn

from torch.nn import functional as F

class CrossFormer(nn.Module):
    def __init__(self, query_dim, hid_dim, q_dim, v_dim, num_heads):
        super(CrossFormer, self).__init__()
        self.query_dim = query_dim
        self.hid_dim = hid_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.Wq = nn.Linear(query_dim, q_dim)
        self.Wk = nn.Linear(hid_dim, q_dim)
        self.Wv = nn.Linear(hid_dim, v_dim)
        self.out_linear = nn.Linear(v_dim, query_dim)

        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim)
        )
        self.norm = nn.RMSNorm(query_dim)
        self.query_norm = nn.RMSNorm(query_dim)
        self.v_norm = nn.RMSNorm(hid_dim)

    def forward(self, query, value):
        query_norm = self.query_norm(query)
        v_norm = self.v_norm(value)

        q = self.Wq(query_norm) # B, 10, q_dim
        k = self.Wk(v_norm) # B, 122, q_dim
        v = self.Wv(v_norm) # B, 122, v_dim

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3) # B, num_heads,  head, 10, head_dim
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3) # B, num_heads,  head, 122, head_dim
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3) # B, num_heads,  head, 122, head_dim



        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        attn_output = attn_output.permute(0,2,1,3).contiguous().view(attn_output.shape[0], attn_output.shape[2], -1) # B, 10, v_dim
        attn_output = self.out_linear(attn_output) # B, 10, query_dim

        attn = query + attn_output
        output = attn + self.norm(self.ffn(attn))

        return output

class Qformer(nn.Module):
    def __init__(self, query_dim, hid_dim, q_dim, v_dim, num_heads, num_layers):
        super(Qformer, self).__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_heads = num_heads


        self.cross_layers = nn.ModuleList()
        self.query_norm_layers = nn.ModuleList()
        self.v_norm_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(122 * 8)


        for i in range(num_layers):
            self.cross_layers.append(CrossFormer(query_dim, hid_dim, q_dim, v_dim, num_heads))

            self.v_norm_layers.append(nn.BatchNorm1d(122 * 8))
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, hid_dim)
                )
            )
    
    def forward(self, query, v):
        # v = v.view(v.shape[0], -1)  # B, 1, L*D
        # v = self.batch_norm(v)
        # v = v.view(v.shape[0], -1, self.hid_dim)  # B, L, D
  
        for i in range(self.num_layers):
            v = v.view(v.shape[0], -1)  # B, 1, L*D
            v = self.v_norm_layers[i](v)
            v = v.view(v.shape[0], -1, self.hid_dim)  # B, L, D
            query = self.cross_layers[i](query, v)

            v = self.mlp_layers[i](v)

        return query


