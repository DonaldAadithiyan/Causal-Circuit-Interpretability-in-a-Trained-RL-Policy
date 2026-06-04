"""
Phase 2 — Collect policy activations then train Top-K SAE.
Activations saved as memory-mapped numpy arrays to avoid RAM overflow.
"""

import sys, os, gc, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from models.topk_sae import TopKSAE
from envs.coin_env import make_env_with_info
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
SAE_CFG_PATH = os.path.join(BASE, "configs/sae.yaml")
OUT_DIR = os.path.join(BASE, "outputs")
ACT_DIR = os.path.join(OUT_DIR, "activations")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


def load_config():
    with open(SAE_CFG_PATH) as f:
        return yaml.safe_load(f)


# ── Activation collection ──────────────────────────────────────────────────

def collect_activations(model, n_steps: int = 100_000):
    """
    Run frozen policy on training distribution, capturing features_extractor output.
    Saves:
      activations.npy    — (N, 256) float32 memmap
      observations.npy   — (N, 64, 64, 3) uint8 memmap
      actions.npy        — (N,) int32
      rewards.npy        — (N,) float32
      goal_pos.npy       — (N, 2) int32
      agent_pos.npy      — (N, 2) int32
    Returns actual number of samples collected.
    """
    os.makedirs(ACT_DIR, exist_ok=True)

    # Pre-allocate memmaps
    features_dim = 256
    obs_shape = (64, 64, 3)

    act_mm = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                       mode="w+", shape=(n_steps, features_dim))
    obs_mm = np.memmap(os.path.join(ACT_DIR, "observations.npy"), dtype=np.uint8,
                       mode="w+", shape=(n_steps, *obs_shape))
    act_arr = np.zeros(n_steps, dtype=np.int32)
    rew_arr = np.zeros(n_steps, dtype=np.float32)
    gpos_arr = np.zeros((n_steps, 2), dtype=np.int32)
    apos_arr = np.zeros((n_steps, 2), dtype=np.int32)

    # Hook to capture features_extractor output
    captured = {}

    def hook_fn(_module, _inp, out):
        captured["feat"] = out.detach().cpu()

    handle = model.policy.features_extractor.register_forward_hook(hook_fn)

    env = make_env_with_info(goal_fixed=True)
    obs, info = env.reset(seed=0)
    idx = 0

    log_entry("Phase 2 — Collecting activations",
              f"- target: {n_steps:,} samples\n- env: CoinCollect training distribution")

    t0 = time.time()
    while idx < n_steps:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=False)

        feat = captured["feat"].squeeze(0).numpy()  # (256,)

        act_mm[idx] = feat
        obs_mm[idx] = obs
        act_arr[idx] = int(action)
        gp = info.get("goal_pos") or (0, 0)
        ap = info.get("agent_pos") or (0, 0)
        gpos_arr[idx] = [int(gp[0]), int(gp[1])]
        apos_arr[idx] = [int(ap[0]), int(ap[1])]

        obs, reward, term, trunc, info = env.step(action)
        rew_arr[idx] = float(reward)
        idx += 1

        if term or trunc:
            obs, info = env.reset()

        if idx % 10000 == 0:
            elapsed = time.time() - t0
            log_entry(f"Phase 2 — {idx:,}/{n_steps:,} samples collected",
                      f"- elapsed: {elapsed/60:.1f} min")

    handle.remove()
    env.close()

    # Flush memmaps, save metadata arrays
    act_mm.flush()
    obs_mm.flush()
    np.save(os.path.join(ACT_DIR, "actions.npy"), act_arr[:idx])
    np.save(os.path.join(ACT_DIR, "rewards.npy"), rew_arr[:idx])
    np.save(os.path.join(ACT_DIR, "goal_pos.npy"), gpos_arr[:idx])
    np.save(os.path.join(ACT_DIR, "agent_pos.npy"), apos_arr[:idx])

    # Save metadata
    meta = {"n_samples": int(idx), "features_dim": features_dim, "obs_shape": list(obs_shape)}
    with open(os.path.join(ACT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_entry("Phase 2 — Collection complete",
              f"- Samples: {idx:,}\n"
              f"- activations saved to {ACT_DIR}\n"
              f"- elapsed: {(time.time()-t0)/60:.1f} min")
    return idx


# ── SAE training ───────────────────────────────────────────────────────────

def train_sae(cfg: dict):
    sae_cfg = cfg["sae"]
    val_split = cfg["validation_split"]

    # Load activation memmap
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_samples"]
    features_dim = meta["features_dim"]

    acts = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                     mode="r", shape=(n, features_dim))

    # Compute mean/std for normalisation
    print("Computing activation statistics...")
    mean = acts[:].mean(axis=0)
    std = acts[:].std(axis=0) + 1e-8
    np.save(os.path.join(ACT_DIR, "act_mean.npy"), mean)
    np.save(os.path.join(ACT_DIR, "act_std.npy"), std)

    # Train/val split
    n_val = int(n * val_split)
    n_train = n - n_val
    idx_perm = np.random.permutation(n)
    train_idx = idx_perm[:n_train]
    val_idx = idx_perm[n_train:]

    train_acts = torch.from_numpy((acts[train_idx] - mean) / std)
    val_acts = torch.from_numpy((acts[val_idx] - mean) / std)

    log_entry("Phase 2 — SAE training start",
              f"- train: {n_train:,}, val: {n_val:,}\n"
              f"- K={sae_cfg['k']}, hidden_factor={sae_cfg['hidden_factor']}\n"
              f"- features_dim={features_dim}, hidden_dim={features_dim * sae_cfg['hidden_factor']}")

    sae = TopKSAE(
        input_dim=features_dim,
        hidden_factor=sae_cfg["hidden_factor"],
        k=sae_cfg["k"],
    ).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=sae_cfg["learning_rate"])
    batch_size = sae_cfg["batch_size"]
    max_epochs = sae_cfg["max_epochs"]
    patience = sae_cfg["patience"]

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        sae.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start: start + batch_size]
            x = train_acts[batch_idx].to(device)
            optimizer.zero_grad()
            x_hat, h = sae(x)
            loss = sae.loss(x, x_hat)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Validation
        sae.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            n_val_batches = 0
            for start in range(0, len(val_acts), batch_size):
                x_val = val_acts[start: start + batch_size].to(device)
                x_hat_val, _ = sae(x_val)
                val_loss_total += sae.loss(x_val, x_hat_val).item()
                n_val_batches += 1
            val_loss = val_loss_total / n_val_batches

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Dead feature count
        with torch.no_grad():
            sample = val_acts[:2000].to(device)
            _, h_sample = sae(sample)
            active_mask = (h_sample > 0).float().sum(0) / 2000
            dead_features = int((active_mask < 0.001).sum().item())

        if epoch % 5 == 0 or epoch == 1:
            log_entry(
                f"Phase 2 — SAE epoch {epoch}/{max_epochs}",
                f"- train_loss: {train_loss:.6f}\n"
                f"- val_loss: {val_loss:.6f}\n"
                f"- dead_features: {dead_features}/{sae.hidden_dim}\n"
                f"- elapsed: {(time.time()-t0)/60:.1f} min",
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            sae_path = os.path.join(CKPT_DIR, "sae_best.pt")
            torch.save({"state_dict": sae.state_dict(),
                        "input_dim": features_dim,
                        "k": sae_cfg["k"],
                        "hidden_factor": sae_cfg["hidden_factor"],
                        "act_mean": mean.tolist(),
                        "act_std": std.tolist()}, sae_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_entry(f"Phase 2 — Early stop at epoch {epoch}",
                          f"- best val_loss: {best_val_loss:.6f}")
                break

    # Final evaluation
    sae.eval()
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Reconstruction loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("SAE Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_loss_curve.png"), dpi=150)
    plt.close()

    # Feature activation frequency
    with torch.no_grad():
        full_val = val_acts[:5000].to(device)
        _, h_full = sae(full_val)
        freq = (h_full > 0).float().mean(0).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.hist(freq, bins=50)
    plt.xlabel("Activation Frequency")
    plt.ylabel("# Features")
    plt.title("SAE Feature Activation Frequency Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_feature_freq.png"), dpi=150)
    plt.close()

    dead = int((freq < 0.001).sum())
    results = {
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "dead_features": dead,
        "total_features": sae.hidden_dim,
        "k": sae_cfg["k"],
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(os.path.join(OUT_DIR, "sae_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log_entry(
        "Phase 2 — SAE training complete",
        f"- Best val_loss: {best_val_loss:.6f}\n"
        f"- Dead features: {dead}/{sae.hidden_dim}\n"
        f"- Plots: sae_loss_curve.png, sae_feature_freq.png\n"
        f"- Checkpoint: {os.path.join(CKPT_DIR, 'sae_best.pt')}",
    )

    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE")
    print(f"Best val MSE:  {best_val_loss:.6f}")
    print(f"Dead features: {dead}/{sae.hidden_dim}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def main():
    cfg = load_config()

    # Load policy
    policy_path = os.path.join(CKPT_DIR, "ppo_final.zip")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}. Run train_policy.py first.")

    model = PPO.load(policy_path, device=str(device))
    model.policy.eval()
    for p in model.policy.parameters():
        p.requires_grad_(False)

    log_entry("Phase 2 START — SAE training",
              f"- Policy loaded from {policy_path}\n"
              f"- Policy frozen (no gradients)\n"
              f"- Device: {device}")

    # Collect activations if not already done
    meta_path = os.path.join(ACT_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n = meta["n_samples"]
        print(f"Found existing activation dataset: {n:,} samples. Skipping collection.")
        log_entry("Phase 2 — Activation dataset found",
                  f"- {n:,} samples at {ACT_DIR}\n- Skipping collection.")
    else:
        n = collect_activations(model, n_steps=cfg["collection"]["n_rollout_steps"])

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    train_sae(cfg)


if __name__ == "__main__":
    main()
