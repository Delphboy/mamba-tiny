"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from scans import selective_scan

from model_args import ModelArgs

class CrossMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.in_proj2 = nn.Linear(args.d_model, args.d_inner, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        assert type(args.dt_rank) is int, "dt_rank is not an integer"
        # self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state, bias=False)
        self.c_proj = nn.Linear(args.d_inner, args.d_state, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, text_input, visual_input):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = text_input.shape

        x_and_res = self.in_proj(text_input)  # shape (b, l, 2 * d_in)
        visual_input = self.in_proj2(visual_input)  # shape (b, l, 2 * d_in)
        (text_input, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        text_input = rearrange(text_input, 'b l d_in -> b d_in l')
        text_input = self.conv1d(text_input)[:, :, :l]
        text_input = rearrange(text_input, 'b d_in l -> b l d_in')

        text_input = F.silu(text_input)

        y = self.ssm(text_input, visual_input)

        y = y * F.silu(res)

        return self.out_proj(y)

    def ssm(self, query, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(query)  # (b, l, dt_rank + 2*n)

        (delta, B) = x_dbl.split(split_size=[self.args.dt_rank, n], dim=-1)  # delta: (b, l, dt_rank). B: (b, l, n)

        C = self.c_proj(x)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        return selective_scan(query, delta, A, B, C, D, mode=self.args.scan_mode)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

class TinyCrossMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualCrossBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper

    def forward(self, text_input, visual_input):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(text_input)
        
        for layer in self.layers:
            x = layer(x, visual_input)
            
        x = self.norm_f(x)
        return self.lm_head(x)

    @staticmethod
    def from_pretrained(pretrained_model_name: str, model=None):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        if model is None:
            config_data = load_config_hf(pretrained_model_name)
            model = TinyCrossMamba(ModelArgs(
                d_model=config_data['d_model'], 
                n_layer=config_data['n_layer'], 
                vocab_size=config_data['vocab_size'], 
            ))
        
        pretrained_dict = load_state_dict_hf(pretrained_model_name)
        model_dict = model.state_dict()
        
        for k, v in pretrained_dict.items():
            k_new = k.replace('backbone.', '')
            if k_new in model_dict and v.size() == model_dict[k_new].size():
                model_dict[k_new] = pretrained_dict[k]
        
        model.load_state_dict(model_dict)
        return model

class ResidualCrossBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = CrossMambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, query, x):
        return self.mixer(self.norm(query), self.norm(x)) + query




class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
