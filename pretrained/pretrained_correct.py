import torch
from collections import OrderedDict

new_state_dict = OrderedDict()
state_dict = torch.load('./mit_b1.pth')
for k, v in state_dict.items():
    k = 'backbone.' + k   # remove prefix backbone.
    # k = k.replace('attn.qkv.weight', 'attn.attn.in_proj_weight')
    # k = k.replace('attn.qkv.bias', 'attn.attn.in_proj_bias')
    # k = k.replace('attn.proj.weight', 'attn.attn.out_proj.weight')
    # k = k.replace('attn.proj.bias', 'attn.attn.out_proj.bias')
    new_state_dict[k] = v

torch.save(new_state_dict, './modified_mit_b1.pth')