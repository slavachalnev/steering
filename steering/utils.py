

import torch
from transformer_lens import HookedTransformer



def get_activation_steering(model: HookedTransformer, hook_point: str, pos_text, neg_text):
    """
    Computes difference in activations between pos and neg at the hook_point.

    Returns:
        torch.Tensor: The steering vector. Shape: [batch_size, sequence_length, hidden_dim]

    Example:
        hp = get_act_name("resid_pre", 4)  # blocks.4.hook_resid_pre
        steering = get_activation_steering(model, hp, pos_text="Anger", neg_text="Calm")
        print(steering.shape)  # torch.Size([1, 3, 768])
    """
    pos = model.to_tokens(pos_text)
    neg = model.to_tokens(neg_text)
    assert pos.shape[-1] == neg.shape[-1]

    _, pos_acts = model.run_with_cache(pos, names_filter=hook_point)
    _, neg_acts = model.run_with_cache(neg, names_filter=hook_point)

    pos_acts = pos_acts[hook_point]
    neg_acts = neg_acts[hook_point]

    steering = pos_acts - neg_acts
    return steering

    

