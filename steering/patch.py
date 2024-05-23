
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformer_lens import HookedTransformer
import transformer_lens.utils as tutils

from steering.eval_utils import evaluate_completions


def patch_resid(resid, hook, steering, c=1, pos=0):
    assert len(steering.shape) == 3 # [batch_size, sequence_length, d_model]

    if pos is None:
        # insert at all positions
        assert steering.shape[1] == 1
        resid[:, :, :] = resid[:, :, :] + c * steering
        return resid

    n_toks = min(resid.shape[1] - pos, steering.shape[1])
    if pos < resid.shape[1]:
        resid[:, pos:n_toks+pos, :] = resid[:, pos:n_toks+pos, :] + c * steering[:, :n_toks, :]
    
    return resid


def get_loss(
        model: HookedTransformer,
        hook_point: str,
        steering_vector=None,
        scales=None,
        ds_name="NeelNanda/c4-code-20k",
        n_batches=20,
        batch_size=8,
        insertion_pos=0,
    ):
    if scales is None:
        scales = [1]
    elif not isinstance(scales, list):
        scales = [scales]
    losses = []

    print(f'loading dataset: {ds_name}')
    data = load_dataset(ds_name, split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=batch_size)
    print('dataset loaded')

    for scale in tqdm(scales):
        total_loss = 0
        for i, batch in enumerate(loader):
            with model.hooks(fwd_hooks=[(hook_point, partial(patch_resid,
                                                            steering=steering_vector,
                                                            c=scale,
                                                            pos=insertion_pos,
                                                            ))]):
                loss = model(batch["tokens"], return_type="loss", prepend_bos=False)
                total_loss += loss.item()
            if i == n_batches:
                break
        losses.append(total_loss / n_batches)
    
    return losses


def generate(
        model: HookedTransformer,
        hook_point: str,
        prompt = "",
        steering_vector=None,
        scale=1,
        n_samples=5,
        max_length=20,
        insertion_pos=0,
    ):
    gen_texts = []
    with model.hooks(fwd_hooks=[(hook_point, partial(patch_resid,
                                                    steering=steering_vector,
                                                    c=scale,
                                                    pos=insertion_pos,
                                                    ))]):
        for _ in tqdm(range(n_samples)):
            output = model.generate(prompt,
                                    prepend_bos=True,
                                    use_past_kv_cache=False,
                                    max_new_tokens=max_length,
                                    verbose=False,
                                    )
            gen_texts.append(output)
    return gen_texts
    


def get_scores_and_losses(
    model: HookedTransformer,
    hook_point: str,
    steering_vector: torch.Tensor,
    prompt: str,
    criterion: str,
    scales: list[float],
    n_samples = 10,
    insertion_pos = 0):

    sae_anger_losses = get_loss(model, hook_point, steering_vector=steering_vector, scales=scales, insertion_pos=insertion_pos)

    mean_scores = []
    all_scores = []
    for scale in scales:
        print("scale", scale)
        gen_texts = generate(model,
                             hook_point,
                             prompt=prompt,
                             steering_vector=steering_vector,
                             scale=scale, n_samples=n_samples,
                             insertion_pos=insertion_pos)
        evals = evaluate_completions(gen_texts, criterion=criterion, prompt=prompt)
        print(gen_texts)
        print(evals)
        scores = [e['score'] for e in evals]
        mean = sum(scores) / len(scores)
        mean_scores.append(mean)
        all_scores.append(scores)
        print("mean", mean)
    
    return mean_scores, all_scores, sae_anger_losses
