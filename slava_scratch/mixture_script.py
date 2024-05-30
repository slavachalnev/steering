import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
from transformer_lens.evals import make_pile_data_loader, evaluate_on_dataset

from functools import partial
from datasets import load_dataset
from tqdm import tqdm

from sae_lens import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens import SparseAutoencoder, ActivationsStore

from steering.eval_utils import evaluate_completions
from steering.utils import text_to_sae_feats, top_activations, normalise_decoder, get_activation_steering
from steering.patch import generate, get_scores_and_losses, patch_resid, get_loss, scores_2d, scores_clamp_2d

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

import plotly.express as px
import plotly.graph_objects as go

torch.set_grad_enabled(False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gemma-2b", device=device)


hp6 = "blocks.6.hook_resid_post"
sae6 = SparseAutoencoder.from_pretrained("gemma-2b-res-jb", hp6)
normalise_decoder(sae6, scale_input=False)
sae6 = sae6.to(device)


feature_ids = [1062, 8406]
feature_descriptions = ["Anger/frustration", "Wedding"]
criterions = ["Text is angry/upset/enraged/distressed or is about anger/hate etc.",
                "Text mentions wedding/marriage/engagement."]
coherence_criterion = "Text is coherent, the grammar is correct."
n_samples = 50
save_dir = "runs/anger_wedding"
os.makedirs(save_dir)

d1 = sae6.W_dec[feature_ids[0]]
d2 = sae6.W_dec[feature_ids[1]]
enc1 = sae6.W_enc[:, feature_ids[0]]
enc2 = sae6.W_enc[:, feature_ids[1]]

scales = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
prompt = "I think"
clamp = False


gen_log_file = f"{save_dir}/gen_log.json"

# log params to .txt file
with open(f"{save_dir}/params.txt", "w") as f:
    f.write(f"prompt: {prompt}\n")
    f.write(f"scales: {scales}\n")
    f.write(f"feature_ids: {feature_ids}\n")
    f.write(f"feature_descriptions: {feature_descriptions}\n")
    f.write(f"criterions: {criterions}\n")
    f.write(f"coherence_criterion: {coherence_criterion}\n")
    f.write(f"n_samples: {n_samples}\n")
    f.write(f"clamp: {clamp}\n")


if clamp:
    scores_1, scores_2, losses, coherence_scores = scores_clamp_2d(
        model,
        hp6,
        # steering_vectors=[d1, d2],
        steering_encoders=[enc1, enc2],
        steering_decoders=[d1, d2],

        prompt=prompt,
        criterions=criterions,
        scales=scales,
        n_samples=n_samples,
        coherence_criterion=coherence_criterion,
        gen_log_file=gen_log_file,
    )
else:
    scores_1, scores_2, losses, coherence_scores = scores_2d(
        model,
        hp6,
        steering_vectors=[d1, d2],
        prompt=prompt,
        criterions=criterions,
        scales=scales,
        n_samples=n_samples,
        coherence_criterion=coherence_criterion,
        gen_log_file=gen_log_file,
    )


# save the scores and losses

torch.save(scores_1, f"{save_dir}/scores_1.pt")
torch.save(scores_2, f"{save_dir}/scores_2.pt")
torch.save(losses, f"{save_dir}/losses.pt")
torch.save(coherence_scores, f"{save_dir}/coherence_scores.pt")

# plot the heatmaps
fig = px.imshow(scores_1, x=scales, y=scales,
          title=f"{feature_descriptions[0]} scores",labels={'x': feature_descriptions[1], 'y': feature_descriptions[0]},
          color_continuous_scale="RdBu", color_continuous_midpoint=0)
fig.write_html(f"{save_dir}/scores_1.html")
fig.write_image(f"{save_dir}/scores_1.png")
        
fig = px.imshow(scores_2, x=scales, y=scales,
            title=f"{feature_descriptions[1]} scores",labels={'x': feature_descriptions[1], 'y': feature_descriptions[0]},
            color_continuous_scale="RdBu", color_continuous_midpoint=0)
fig.write_html(f"{save_dir}/scores_2.html")
fig.write_image(f"{save_dir}/scores_2.png")

fig = px.imshow(coherence_scores, x=scales, y=scales,
            title="Coherence scores",labels={'x': feature_descriptions[1], 'y': feature_descriptions[0]},
            color_continuous_scale="RdBu", color_continuous_midpoint=0)
fig.write_html(f"{save_dir}/coherence_scores.html")
fig.write_image(f"{save_dir}/coherence_scores.png")

fig = px.imshow(losses, x=scales, y=scales,
            title="Losses",labels={'x': feature_descriptions[1], 'y': feature_descriptions[0]},
            color_continuous_scale="RdBu", color_continuous_midpoint=0)
fig.write_html(f"{save_dir}/losses.html")
fig.write_image(f"{save_dir}/losses.png")



