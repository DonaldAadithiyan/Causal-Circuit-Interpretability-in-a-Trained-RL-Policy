"""
Experiment 2, Phase 2 — EAP-based G_live computation.
Provides:
  - compute_eap_weights(): gradient × activation attribution per step
  - validate_eap_vs_patching(): Pearson r between EAP and patching
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def compute_eap_weights(
    acts_norm: torch.Tensor,   # (batch, 256) normalized activations on device
    sae,
    model,
    std_t: torch.Tensor,
    mean_t: torch.Tensor,
) -> torch.Tensor:
    """
    EAP weight_i = |h_i × ∂(selected_logit)/∂h_i|
    Returns (batch, hidden_dim) tensor of causal weights (no grad).
    """
    with torch.no_grad():
        h_pre = sae.encoder(acts_norm)
        h = sae.top_k_gate(h_pre)

    # Detach h so gradient flows from logits → decoder → h (not through encoder)
    h_for_grad = h.detach().requires_grad_(True)

    recon_norm = sae.decoder(h_for_grad)
    recon_raw = recon_norm * std_t + mean_t
    logits = model.policy.action_net(recon_raw)

    selected_action = logits.argmax(dim=-1)
    selected_logit = logits.gather(1, selected_action.unsqueeze(1)).squeeze(1)

    grads = torch.autograd.grad(selected_logit.sum(), h_for_grad)[0]

    return (h.detach() * grads.detach()).abs()  # (batch, hidden_dim)


def validate_eap_vs_patching(acts_norm: torch.Tensor, sae, model,
                              std_t: torch.Tensor, mean_t: torch.Tensor,
                              top32: list) -> float:
    """
    Compute Pearson r between mean EAP weight and patching KL for each feature in top32.
    Returns correlation coefficient in [-1, 1].
    """
    eap = compute_eap_weights(acts_norm, sae, model, std_t, mean_t)
    eap_mean = eap.mean(0).detach().cpu().numpy()
    eap_top32 = np.array([eap_mean[f] for f in top32])

    with torch.no_grad():
        _, h_b = sae(acts_norm)
        logits_b = model.policy.action_net(sae.decode(h_b) * std_t + mean_t)
        probs_b = F.softmax(logits_b, dim=-1)

    kl_vals = []
    for feat in top32:
        with torch.no_grad():
            h_p = h_b.clone()
            h_p[:, feat] = 0.0
            logits_p = model.policy.action_net(sae.decode(h_p) * std_t + mean_t)
            probs_p = F.softmax(logits_p, dim=-1)
            p = probs_b + 1e-8
            q = probs_p + 1e-8
            kl_vals.append((p * torch.log(p / q)).sum(-1).mean().item())

    kl_arr = np.array(kl_vals)
    if eap_top32.std() < 1e-10 or kl_arr.std() < 1e-10:
        return 0.0
    r, _ = pearsonr(eap_top32, kl_arr)
    return float(r)


if __name__ == "__main__":
    # Quick smoke test
    print("compute_glive module OK")
