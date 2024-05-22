import os
import sys
sys.path.append(os.path.abspath('..'))

from tqdm import tqdm

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

import torch
from transformer_lens import HookedTransformer
from sae_lens import SparseAutoencoder, ActivationsStore

from steering.utils import normalise_decoder

torch.set_grad_enabled(False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gemma-2b", device='cpu')


hp_6 = "blocks.6.hook_resid_post"
sae_6 = SparseAutoencoder.from_pretrained("gemma-2b-res-jb", hp_6)
normalise_decoder(sae_6)
activation_store = ActivationsStore.from_config(model, sae_6.cfg)


def get_tokens(
    activation_store,
    n_batches_to_sample_from: int = 2**15
):
    all_tokens_list = []
    pbar = tqdm(range(n_batches_to_sample_from))
    for _ in pbar:
        batch_tokens = activation_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens


all_tokens = get_tokens(activation_store).to(device)

model = model.to(device)
sae_6 = sae_6.to(device)


n_features = sae_6.cfg.d_sae

test_feature_idx_gpt = range(n_features)
bs = 4

feature_vis_config_gpt = SaeVisConfig(
    hook_point=hp_6,
    features=test_feature_idx_gpt,
    batch_size=bs,
    minibatch_size_tokens=128,
    verbose=True,
)

with torch.inference_mode():
    sae_vis_data_gpt = SaeVisData.create(
        encoder=sae_6,
        model=model,
        tokens=all_tokens,  # type: ignore
        cfg=feature_vis_config_gpt,
    )


vis_dir = "feature_vis"
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

for idx, feature in enumerate(test_feature_idx_gpt):
    if sae_vis_data_gpt.feature_stats.max[idx] == 0:
        continue
    filename = os.path.join(vis_dir, f"{feature}_feature_vis.html")
    try:
        sae_vis_data_gpt.save_feature_centric_vis(filename, feature)
    except ZeroDivisionError:
        print(f"Skipped feature {feature} due to ZeroDivisionError.")
