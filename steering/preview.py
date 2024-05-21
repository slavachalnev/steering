import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
from transformer_lens.evals import make_pile_data_loader, evaluate_on_dataset


from steering.visualization import Table


def get_indices(tensor, values):
    """
    Get the indices of the given values in the tensor.

    Args:
    tensor (torch.Tensor): The tensor to search in.
    values (list): The list of values to find in the tensor.

    Returns:
    list: A list of indices where the values are found in the tensor.
    """
    indices = []
    for value in values:
        idx = torch.nonzero(torch.eq(tensor, value)).squeeze()
        if idx.numel() > 0:  # Check if there are any indices found
            indices.append(idx.item())
    return indices


def preview_next_step(
    model: HookedTransformer, prompt: str, fwd_hooks=[], watch_logits: list = []
):
    """
    Runs a single forward pass over the model and logit information

    You can pass in any fwd_hooks that you want to intervene with

    Args:
    watch_logits: a list of vocab index positions that you want to watch
    """
    logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks, prepend_bos=True)

    # get logit information
    ranked_logits = logits[0, -1].topk(logits.shape[-1])
    indices = ranked_logits.indices
    values = ranked_logits.values

    # get top 20 logits
    top_values = [value.item() for value in values[:20]]

    top_indices = indices[:20]
    top_tokens = model.tokenizer.batch_decode(top_indices)
    positions = list(range(1, 21))
    Table(
        "Top Tokens",
        ["Positions", "Token", "Act"],
        zip(positions, top_tokens, top_values),
    )

    if len(watch_logits) > 0:
        watch_positions = get_indices(indices, watch_logits)
        watch_values = [value.item() for value in values[watch_positions]]

        watch_indices = indices[watch_positions]
        watch_tokens = model.tokenizer.batch_decode(watch_indices)

        Table(
            "Watch Tokens",
            ["Positions", "Token", "Act"],
            zip(watch_positions, watch_tokens, watch_values),
        )


def generate(
    model: HookedTransformer, prompt: str, fwd_hooks=[], n_samples=5, max_length=20
):
    gen_texts = []
    with model.hooks(fwd_hooks=fwd_hooks):
        for _ in tqdm(range(n_samples)):
            output = model.generate(
                prompt,
                prepend_bos=True,
                use_past_kv_cache=False,
                max_new_tokens=max_length,
                verbose=False,
            )
            gen_texts.append(output)
    return gen_texts
