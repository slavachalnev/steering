
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformer_lens import HookedTransformer
import transformer_lens.utils as tutils

from steering.eval_utils import evaluate_completions


def patch_resid(resid, hook, steering, c=1, pos=None):
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
        n_batches=50,
        batch_size=8,
        insertion_pos=None,
        patch_fn=None,
    ):
    if patch_fn is None:
        patch_fn = patch_resid

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
            with model.hooks(fwd_hooks=[(hook_point, partial(patch_fn,
                                                            steering=steering_vector,
                                                            c=scale,
                                                            pos=insertion_pos,
                                                            ))]):
                loss = model(batch["tokens"], return_type="loss", prepend_bos=False)
                total_loss += loss.item()
            if i >= n_batches-1:
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
        max_length=25,
        insertion_pos=0,
        top_k=50,
        top_p=0.3,
        patch_fn=None,
    ):
    if patch_fn is None:
        patch_fn = patch_resid

    gen_texts = []
    with model.hooks(fwd_hooks=[(hook_point, partial(patch_fn,
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
                                    top_k=top_k,
                                    top_p=top_p,
                                    )
            gen_texts.append(output)
    return gen_texts
    

def contains_word(text, word_list):
    for word in word_list:
        if word.lower() in text.lower():
            return True
    return False


def get_scores_and_losses(
    model: HookedTransformer,
    hook_point: str,
    steering_vector: torch.Tensor,
    prompt: str,
    criterion: str,
    scales: list[float],
    n_samples=10,
    insertion_pos=None,
    explanations=True,
    word_list=None,
    top_k=50,
):

    sae_anger_losses = get_loss(model, hook_point, steering_vector=steering_vector, scales=scales, insertion_pos=insertion_pos)

    mean_scores = []
    all_scores = []
    word_probabilities = []

    for scale in scales:
        print("scale", scale)
        gen_texts = generate(model,
                             hook_point,
                             prompt=prompt,
                             steering_vector=steering_vector,
                             scale=scale,
                             n_samples=n_samples,
                             insertion_pos=insertion_pos,
                             top_k=top_k,
                             )
        evals = evaluate_completions(gen_texts, criterion=criterion, prompt=prompt, verbose=explanations)
        print(gen_texts)
        print(evals)
        scores = [e['score'] for e in evals]
        mean = sum(scores) / len(scores)
        mean_scores.append(mean)
        all_scores.append(scores)
        print("mean", mean)
        
        if word_list is not None:
            count = 0
            for text in gen_texts:
                if contains_word(text, word_list):
                    count += 1
            word_prob = count / n_samples
            word_probabilities.append(word_prob)
            print("word probability", word_prob)

    if word_list is not None:
        return mean_scores, all_scores, sae_anger_losses, word_probabilities
    else:
        return mean_scores, all_scores, sae_anger_losses


@torch.no_grad()
def grid_losses(
        model: HookedTransformer,
        hook_point: str,
        vector_grid: torch.Tensor,
        ds_name="NeelNanda/c4-code-20k",
        n_batches=50,
        batch_size=8,
        insertion_pos=None,
    ):
    """
    Used in scores_2d
    """
    loss_grid = torch.zeros((vector_grid.shape[0], vector_grid.shape[1]), device=vector_grid.device)

    print(f'loading dataset: {ds_name}')
    data = load_dataset(ds_name, split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=batch_size)
    print('dataset loaded')

    for i in tqdm(range(vector_grid.shape[0])):
        for j in range(vector_grid.shape[1]):
            steering = vector_grid[i, j]
            steering = steering[None, None, :]
            total_loss = 0
            for batch_idx, batch in enumerate(loader):
                with model.hooks(fwd_hooks=[(hook_point, partial(patch_resid,
                                                                steering=steering,
                                                                c=1,
                                                                pos=insertion_pos,
                                                                ))]):
                    loss = model(batch["tokens"], return_type="loss", prepend_bos=False)
                    total_loss += loss.item()
                if batch_idx >= n_batches-1:
                    break
            loss_grid[i, j] = total_loss / n_batches
    return loss_grid


@torch.no_grad()
def scores_2d(
    model: HookedTransformer,
    hook_point: str,
    steering_vectors: list[torch.Tensor],
    prompt: str,
    criterions: list[str],
    scales: list[float],
    n_samples=10,
    insertion_pos=None,
    top_k=50,
    coherence_criterion="Text is coherent, the grammar is correct.",
    ):
    assert len(steering_vectors) == len(criterions) == 2

    v1, v2 = steering_vectors
    device = v1.device

    vector_grid = torch.zeros((len(scales), len(scales), v1.shape[-1]), device=device)
    for i, s1 in enumerate(scales):
        for j, s2 in enumerate(scales):
            vector_grid[i, j] = v1 * s1 + v2 * s2

    loss_grid = grid_losses(model, hook_point, vector_grid, insertion_pos=insertion_pos)
    print("loss grid")
    print(loss_grid)

    score_grid_1 = torch.zeros_like(loss_grid, device=device)
    score_grid_2 = torch.zeros_like(loss_grid, device=device)
    coherence_grid = torch.zeros_like(loss_grid, device=device)

    for i in range(len(scales)):
        for j in range(len(scales)):
            print(f'evaluating ({i}, {j})')
            steering_vector = vector_grid[i, j]
            gen_texts = generate(model,
                                hook_point,
                                prompt=prompt,
                                steering_vector=steering_vector[None, None, :],
                                scale=1,
                                n_samples=n_samples,
                                insertion_pos=insertion_pos,
                                top_k=top_k,
                                )

            eval_1 = evaluate_completions(gen_texts, criterion=criterions[0], prompt=prompt, verbose=False)
            eval_2 = evaluate_completions(gen_texts, criterion=criterions[1], prompt=prompt, verbose=False)
            coherence = evaluate_completions(gen_texts, criterion=coherence_criterion, prompt=prompt, verbose=False)
            print(gen_texts)

            scores_1 = [e['score'] for e in eval_1]
            scores_2 = [e['score'] for e in eval_2]
            coherence_scores = [e['score'] for e in coherence]
            mean_1 = sum(scores_1) / len(scores_1)
            mean_2 = sum(scores_2) / len(scores_2)
            mean_coherence = sum(coherence_scores) / len(coherence_scores)
            score_grid_1[i, j] = mean_1
            score_grid_2[i, j] = mean_2
            coherence_grid[i, j] = mean_coherence
            print(criterions[0], mean_1)
            print(criterions[1], mean_2)
            print(coherence_criterion, mean_coherence)
        
    return score_grid_1.cpu(), score_grid_2.cpu(), loss_grid.cpu(), coherence_grid.cpu()
