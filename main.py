import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import Mamba
from mamba import MambaBlock
from cross_mamba import CrossMambaBlock
from model_args import ModelArgs

supported_models = [
    'state-spaces/mamba-2.8b-slimpj',
    'state-spaces/mamba-2.8b',
    'state-spaces/mamba-1.4b',
    'state-spaces/mamba-790m',
    'state-spaces/mamba-370m',
    'state-spaces/mamba-130m',
]
model = Mamba.from_pretrained(supported_models[4])
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')



def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40):
    model.eval()

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]

        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions

if __name__ == "__main__":
    B, X, Y, D = 8, 15, 44, 512

    args = ModelArgs(d_model=D, 
                     n_layer=6,
                     vocab_size=10000)

    mamba_block = MambaBlock(args)

    input_sequence = torch.rand([B, X, D], dtype=torch.float)
    print("input sequence:", input_sequence.shape)

    mamba_output = mamba_block(input_sequence)
    print("mamba_output:", mamba_output.shape)
    assert input_sequence.shape == mamba_output.shape, "Mamba's output shape does not match the input shape"

    print()
    print("*"*80)
    print()

    args = ModelArgs(d_model=D, 
                     n_layer=6,
                     vocab_size=10000,
                     scan_mode='modified')

    cross_mamba_block = CrossMambaBlock(args)
    input_sequence = torch.rand([B, X, D], dtype=torch.float)
    print("input sequence:", input_sequence.shape)
    image_sequence = torch.rand([B, X, D], dtype=torch.float)
    print("image sequence:", image_sequence.shape)

    cross_mamba_output = cross_mamba_block(input_sequence, image_sequence)
    print("cross_mamba_output:", cross_mamba_output.shape)
    assert input_sequence.shape == cross_mamba_output.shape, "Cross Mamba's output shape does not match the query shape"

    print()
    print("*"*80)
    print()

    args = ModelArgs(d_model=D, 
                     n_layer=6,
                     vocab_size=10000,
                     scan_mode='modified')

    cross_mamba_block = CrossMambaBlock(args)
    input_sequence = torch.rand([B, X, D], dtype=torch.float)
    print("input sequence:", input_sequence.shape)
    image_sequence = torch.rand([B, Y, D], dtype=torch.float)
    print("image sequence:", image_sequence.shape)

    cross_mamba_output = cross_mamba_block(input_sequence, image_sequence)
    print("cross_mamba_output:", cross_mamba_output.shape)
    assert input_sequence.shape == cross_mamba_output.shape, "Cross Mamba's output shape does not match the query shape"
