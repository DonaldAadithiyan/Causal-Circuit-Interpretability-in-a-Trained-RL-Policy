"""
Retrain SAE with feature resampling (nudging).
Uses existing 100k activation dataset — no new rollouts needed.
Target: < 200 dead features out of 512.
Config: hidden_factor=2 (512), K=32, resample every 100 batches.
"""

import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.topk_sae_v2 import TopKSAEv2
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
ACT_DIR = os.path.join(BASE, "outputs/activations")
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
PLOT_DIR = os.path.join(BASE, "outputs/plots")

# Hypers
K = 32
HIDDEN_FACTOR = 1.5   # 384 hidden units (1.5× input) — sized to keep dead features < 200
LR = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 80
PATIENCE = 8
RESAMPLE_EVERY = 50    # resample dead features every N batches
RESAMPLE_THRESHOLD = 150
VAL_SPLIT = 0.1


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load existing activation dataset
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_samples"]
    dim = meta["features_dim"]

    acts = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                     mode="r", shape=(n, dim))

    mean = acts[:].mean(axis=0)
    std = acts[:].std(axis=0) + 1e-8
    np.save(os.path.join(ACT_DIR, "act_mean_v2.npy"), mean)
    np.save(os.path.join(ACT_DIR, "act_std_v2.npy"), std)

    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_acts = torch.from_numpy(((acts[train_idx] - mean) / std).astype(np.float32))
    val_acts   = torch.from_numpy(((acts[val_idx]   - mean) / std).astype(np.float32))

    log_entry("[SAEv2] Retraining SAE with resampling",
              f"- hidden_factor={HIDDEN_FACTOR} → hidden_dim={dim*HIDDEN_FACTOR}\n"
              f"- K={K}, resample_every={RESAMPLE_EVERY} batches\n"
              f"- train={n_train:,}, val={n_val:,}")

    sae = TopKSAEv2(input_dim=dim, hidden_factor=HIDDEN_FACTOR, k=K,
                    resample_threshold=RESAMPLE_THRESHOLD).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    best_val = float("inf")
    patience_ctr = 0
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        sae.train()
        perm_e = torch.randperm(n_train)
        epoch_loss, n_batches = 0.0, 0
        batch_num = 0

        epoch_resampled = 0
        for start in range(0, n_train, BATCH_SIZE):
            batch = train_acts[perm_e[start: start + BATCH_SIZE]].to(device)
            optimizer.zero_grad()
            x_hat, _h = sae(batch)
            loss = sae.loss(batch, x_hat)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            # Resample dead features periodically (skip the last few epochs to let them settle)
            if batch_num % RESAMPLE_EVERY == 0 and epoch <= MAX_EPOCHS - 10:
                epoch_resampled += sae.resample_dead_features(batch, optimizer)

            epoch_loss += loss.item()
            n_batches += 1
            batch_num += 1

        train_loss = epoch_loss / n_batches

        sae.eval()
        val_loss_total, n_vb = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_acts), BATCH_SIZE):
                xv = val_acts[start: start + BATCH_SIZE].to(device)
                xv_hat, _ = sae(xv)
                val_loss_total += sae.loss(xv, xv_hat).item()
                n_vb += 1
        val_loss = val_loss_total / n_vb

        # Dead feature count
        with torch.no_grad():
            sample = val_acts[:2000].to(device)
            _, h_s = sae(sample)
            freq = (h_s > 0).float().mean(0).cpu().numpy()
            dead = int((freq < 0.001).sum())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            log_entry(f"[SAEv2] Epoch {epoch}/{MAX_EPOCHS}",
                      f"- train_loss: {train_loss:.6f}\n"
                      f"- val_loss: {val_loss:.6f}\n"
                      f"- dead_features: {dead}/{sae.hidden_dim}\n"
                      f"- elapsed: {(time.time()-t0)/60:.1f} min")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_ctr = 0
            torch.save({
                "state_dict": sae.state_dict(),
                "input_dim": dim,
                "k": K,
                "hidden_factor": HIDDEN_FACTOR,
                "act_mean": mean.tolist(),
                "act_std": std.tolist(),
                "dead_features": dead,
                "val_loss": best_val,
            }, os.path.join(CKPT_DIR, "sae_v2_best.pt"))
        else:
            patience_ctr += 1
            # Don't early-stop before dead features drop below 20%
            if patience_ctr >= PATIENCE and dead < sae.hidden_dim * 0.20:
                log_entry(f"[SAEv2] Early stop at epoch {epoch}",
                          f"- best val_loss: {best_val:.6f}, dead: {dead}")
                break

    # Final evaluation
    with torch.no_grad():
        full_val = val_acts[:5000].to(device)
        _, h_full = sae(full_val)
        freq_full = (h_full > 0).float().mean(0).cpu().numpy()
        dead_final = int((freq_full < 0.001).sum())

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"SAEv2 Loss (best val={best_val:.6f}, dead={dead_final}/{sae.hidden_dim})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_v2_loss.png"), dpi=150)
    plt.close()

    # Plot freq dist
    plt.figure(figsize=(8, 4))
    plt.hist(freq_full, bins=50)
    plt.xlabel("Activation Frequency"); plt.ylabel("# Features")
    plt.title("SAEv2 Feature Activation Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_v2_freq.png"), dpi=150)
    plt.close()

    log_entry("[SAEv2] Retraining complete",
              f"- best val_loss: {best_val:.6f}\n"
              f"- dead_features: {dead_final}/{sae.hidden_dim}\n"
              f"- elapsed: {(time.time()-t0)/60:.1f} min")

    print(f"\n{'='*60}")
    print(f"SAE v2 RETRAINING COMPLETE")
    print(f"Best val MSE:  {best_val:.6f}")
    print(f"Dead features: {dead_final}/{sae.hidden_dim}  ({dead_final/sae.hidden_dim*100:.1f}%)")
    print(f"Elapsed:       {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return dead_final, best_val


if __name__ == "__main__":
    main()
