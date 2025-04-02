import torch

from tiny_mamba import TinyMamba
from cross_mamba import TinyCrossMamba
from model_args import ModelArgs

supported_models = [
    'state-spaces/mamba-2.8b-slimpj',
    'state-spaces/mamba-2.8b',
    'state-spaces/mamba-1.4b',
    'state-spaces/mamba-790m',
    'state-spaces/mamba-370m',
    'state-spaces/mamba-130m',
]

if __name__ == "__main__":
    B, X, Y, D = 8, 15, 44, 512

    args = ModelArgs(d_model=D, 
                     n_layer=6,
                     vocab_size=10000)

    mamba_block = TinyMamba(args)

    input_sequence = torch.randint(0, 100, [B, X], dtype=torch.int)
    print("input sequence:", input_sequence.shape)

    mamba_output = mamba_block(input_sequence)
    print("mamba_output:", mamba_output.shape)

    print()
    print("*"*80)
    print()

    args = ModelArgs(d_model=D, 
                     n_layer=6,
                     vocab_size=10000,
                     scan_mode='modified')

    cross_mamba_block = TinyCrossMamba(args)
    input_sequence = torch.randint(0, 100, [B, X], dtype=torch.int)
    print("input sequence:", input_sequence.shape)
    image_sequence = torch.rand([B, X, D], dtype=torch.float)
    print("image sequence:", image_sequence.shape)

    cross_mamba_output = cross_mamba_block(input_sequence, image_sequence)
    print("cross_mamba_output:", cross_mamba_output.shape)

    print()
    print("*"*80)
    print()

    args = ModelArgs(d_model=D, 
                     n_layer=6,
                     vocab_size=10000,
                     scan_mode='modified')

    cross_mamba_block = TinyCrossMamba(args)
    input_sequence = torch.randint(0, 100, [B, X], dtype=torch.int)
    print("input sequence:", input_sequence.shape)
    image_sequence = torch.rand([B, Y, D], dtype=torch.float)
    print("image sequence:", image_sequence.shape)

    cross_mamba_output = cross_mamba_block(input_sequence, image_sequence)
    print("cross_mamba_output:", cross_mamba_output.shape)
