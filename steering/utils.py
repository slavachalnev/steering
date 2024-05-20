###
# These utils are mostly for steering vector creation. 
###

from typing import List, Tuple
import torch
from transformer_lens import HookedTransformer
from sae_lens import SparseAutoencoder


@torch.no_grad()
def get_activation_steering(
        model: HookedTransformer,
        hook_point: str,
        pos_text,
        neg_text,
    ):
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


@torch.no_grad()
def get_sae_diff_steering(model: HookedTransformer, hook_point: str, sae: SparseAutoencoder, pos_text, neg_text):
    """
    Computes difference in SAEs between pos and neg at the hook_point.
    """
    pos = model.to_tokens(pos_text)
    neg = model.to_tokens(neg_text)
    assert pos.shape[-1] == neg.shape[-1]

    _, pos_acts = model.run_with_cache(pos, names_filter=hook_point)
    _, neg_acts = model.run_with_cache(neg, names_filter=hook_point)

    pos_acts = pos_acts[hook_point]
    neg_acts = neg_acts[hook_point]

    pos_out = sae(pos_acts)
    pos_sae_out, pos_sae_acts = pos_out.sae_out, pos_out.feature_acts
    neg_out = sae(neg_acts)
    neg_sae_out, neg_sae_acts = neg_out.sae_out, neg_out.feature_acts

    steering = pos_sae_out - neg_sae_out

    return steering, pos_sae_acts, neg_sae_acts

@torch.no_grad()
def remove_sae_feats(
        steering_vec: torch.Tensor,
        sae: SparseAutoencoder,
        feats_to_remove: List[Tuple[int, int, float]],
    ):
    """
    Removes specified features from the steering vector for specified positions.
    
    Args:
        steering_vec (torch.Tensor): The steering vector. Shape: [batch_size, sequence_length, hidden_dim]
        sae (SparseAutoencoder): The sparse autoencoder model.
        feats_to_remove (List[Tuple[int, int, float]] or List[Tuple[int, float]]): A list of tuples where each tuple contains:
            - position (int, optional): The sequence position to remove the feature from.
            - feature index (int): The index of the feature to remove.
            - value (float): The value to remove for the specified feature.
    
    Returns:
        torch.Tensor: The modified steering vector with specified features removed.
    """
    steering_vec = steering_vec.detach().clone()

    for feat_info in feats_to_remove:
        if len(feat_info) == 3:
            pos, feat_idx, val = feat_info
            vec_to_remove = sae.W_dec[feat_idx, :] * val
            steering_vec[:, pos, :] -= vec_to_remove
        elif len(feat_info) == 2:
            feat_idx, val = feat_info
            vec_to_remove = sae.W_dec[feat_idx, :] * val
            steering_vec -= vec_to_remove
    
    return steering_vec


@torch.no_grad()
def text_to_sae_feats(
        model: HookedTransformer,
        sae: SparseAutoencoder,
        hook_point: str,
        text: str,
    ):
    """
    Converts text to SAE features.

    Returns:
        torch.Tensor: SAE activations. Shape: [batch_size, sequence_len, d_sae]
    """

    _, acts = model.run_with_cache(text, names_filter=hook_point)
    acts = acts[hook_point]

    all_sae_acts = []
    for batch in acts:
        sae_acts = sae(batch).feature_acts
        all_sae_acts.append(sae_acts)
    
    return torch.stack(all_sae_acts, dim=0)


@torch.no_grad()
def top_activations(activations: torch.Tensor, top_k: int):
    """
    Returns the top_k activations for each position in the sequence.
    """
    top_v, top_i = torch.topk(activations, top_k, dim=-1)

    return top_v, top_i
