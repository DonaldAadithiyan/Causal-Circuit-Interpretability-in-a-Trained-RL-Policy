"""
attribution_circuit.py

Offline attribution step — run ONCE after training to discover which SAE features
causally drive goal vs shortcut behaviour.

APPROACH
--------
Since the policy uses net_arch=[] (no extra MLP between the feature extractor and
action head), the post-SAE computation is:

    action_logits = W_action @ (W_dec @ h + b_dec) + b_action

This is LINEAR in h.  Attribution patching is therefore exact:

    IE(feature_f; patch_from_hack_to_clean) =
        (W_action @ W_dec[:, f]) * (h_clean[f] - h_hack[f])

IE is a vector over actions.  We reduce to a scalar with the L2 norm so as not
to commit to any particular goal-vs-shortcut action pair upfront.

CLASSIFICATION
--------------
  delta_h[f] = mean(h_hack[:, f]) - mean(h_clean[:, f])

  Goal features  : delta_h[f] < -threshold  (active in clean, suppressed in hack)
  Hack features  : delta_h[f] > +threshold  (active in hack, suppressed in clean)
  IE score       : || (W_action @ W_dec[:, f]) || * |delta_h[f]|
                   — how much does feature f's shift causally change the action logits?

OUTPUT
------
  outputs/attribution_circuit.json
    {
      "goal_features":  [list of feature indices, ranked by IE descending],
      "hack_features":  [list of feature indices, ranked by IE descending],
      "ie_scores":      {str(f): float, ...},    -- all 384 features
      "delta_h":        {str(f): float, ...},    -- mean(hack) - mean(clean)
      "circuit_coeff_norm": {str(f): float, ...},-- ||C[:, f]|| = ||W_action @ W_dec[:, f]||
      "n_clean":  int,
      "n_hack":   int,
    }

USAGE
-----
    python experiment/attribution_circuit.py

    # In code:
    from attribution_circuit import AttributionCircuit
    circuit = AttributionCircuit.discover(policy_path, sae_path, h_clean, h_hack)
    circuit.save("outputs/attribution_circuit.json")

    circuit = AttributionCircuit.load("outputs/attribution_circuit.json")
    circuit.goal_features   # list[int]
    circuit.hack_features   # list[int]
    circuit.ie_scores       # {feature_id: float}
"""

import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.topk_sae_v2 import TopKSAEv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = os.path.dirname(__file__)
POLICY_PATH = os.path.join(BASE, "outputs/checkpoints/ppo_final.zip")
SAE_PATH    = os.path.join(BASE, "outputs/q5_rescore/hack_sae.pt")
EP_DIR      = os.path.join(BASE, "outputs/contrastive/episodes")
OUT_PATH    = os.path.join(BASE, "outputs/attribution_circuit.json")

# Number of top features to include in each role list
TOP_K_GOAL = 8
TOP_K_HACK = 8

# Activation-difference threshold to exclude near-zero features
DELTA_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Load policy action-head weights
# ---------------------------------------------------------------------------

def _load_action_head(policy_path: str) -> np.ndarray:
    """
    Load SB3 PPO from zip and extract the action-net linear layer weights.

    Returns W_action: (n_actions, 256)
    """
    from stable_baselines3 import PPO
    model = PPO.load(policy_path, device="cpu")
    W = model.policy.action_net.weight.detach().cpu().numpy()   # (n_actions, 256)
    return W


def _load_sae_decoder(sae_path: str) -> np.ndarray:
    """
    Load TopK SAE and extract decoder weight matrix.

    Returns W_dec: (256, n_features)
    """
    ckpt = torch.load(sae_path, map_location="cpu")
    sae  = TopKSAEv2(input_dim=256, hidden_factor=1.5, k=32, resample_threshold=150)
    sae.load_state_dict(ckpt["state_dict"])
    W = sae.decoder.weight.detach().numpy()   # (256, n_features)
    return W


# ---------------------------------------------------------------------------
# Compute mean activation vectors from episode arrays
# ---------------------------------------------------------------------------

def _mean_activations(h_list: list) -> np.ndarray:
    """
    h_list: list of (n_steps, n_features) arrays
    Returns mean activation per feature across all steps and episodes.
    """
    total = None
    n     = 0
    for h in h_list:
        total = h.sum(axis=0) if total is None else total + h.sum(axis=0)
        n    += h.shape[0]
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Core attribution
# ---------------------------------------------------------------------------

def _compute_circuit_attribution(
    W_action: np.ndarray,    # (n_actions, 256)
    W_dec:    np.ndarray,    # (256, n_features)
    h_clean:  list,          # list of (n_steps, n_features) arrays
    h_hack:   list,          # list of (n_steps, n_features) arrays
) -> dict:
    """
    Compute per-feature IE scores and classify roles.

    Returns a dict with all attribution data needed to build AttributionCircuit.
    """
    n_features = W_dec.shape[1]

    # Circuit coefficient matrix: C[a, f] = how much does feature f
    # affect action logit a?  Linear because net_arch=[].
    C = W_action @ W_dec                             # (n_actions, n_features)
    C_norm = np.linalg.norm(C, axis=0)              # (n_features,) per-feature magnitude

    # Mean activations per feature
    mu_clean = _mean_activations(h_clean)            # (n_features,)
    mu_hack  = _mean_activations(h_hack)             # (n_features,)
    delta_h  = mu_hack - mu_clean                    # positive = more active in hack

    # IE score: how much does this feature's shift causally move action logits?
    # IE[f] = ||C[:, f]|| * |delta_h[f]|
    ie_scores = C_norm * np.abs(delta_h)             # (n_features,)

    # Classify: goal features have delta_h < 0 (suppressed in hack)
    #           hack features have delta_h > 0 (enhanced in hack)
    goal_mask = delta_h < -DELTA_THRESHOLD
    hack_mask = delta_h > +DELTA_THRESHOLD

    # Rank each group by IE descending
    def _top_k(mask, k):
        idxs = np.where(mask)[0]
        ranked = idxs[np.argsort(ie_scores[idxs])[::-1]]
        return ranked[:k].tolist()

    goal_features = _top_k(goal_mask, TOP_K_GOAL)
    hack_features = _top_k(hack_mask, TOP_K_HACK)

    return {
        "goal_features":       goal_features,
        "hack_features":       hack_features,
        "ie_scores":           {str(f): float(ie_scores[f])    for f in range(n_features)},
        "delta_h":             {str(f): float(delta_h[f])      for f in range(n_features)},
        "circuit_coeff_norm":  {str(f): float(C_norm[f])       for f in range(n_features)},
        "n_clean":             len(h_clean),
        "n_hack":              len(h_hack),
    }


# ---------------------------------------------------------------------------
# AttributionCircuit class
# ---------------------------------------------------------------------------

class AttributionCircuit:
    """
    Holds the offline-computed causal circuit for one frozen policy.

    goal_features : list[int]  — features suppressed during hacking, ranked by IE
    hack_features : list[int]  — features enhanced during hacking, ranked by IE
    ie_scores     : dict[int → float]  — causal importance of every feature
    """

    def __init__(
        self,
        goal_features: list,
        hack_features: list,
        ie_scores:     dict,   # int → float
        delta_h:       dict,   # int → float
        circuit_coeff_norm: dict,  # int → float
        n_clean: int = 0,
        n_hack:  int = 0,
    ):
        self.goal_features       = goal_features
        self.hack_features       = hack_features
        self.ie_scores           = ie_scores
        self.delta_h             = delta_h
        self.circuit_coeff_norm  = circuit_coeff_norm
        self.n_clean             = n_clean
        self.n_hack              = n_hack

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def discover(
        cls,
        policy_path: str,
        sae_path:    str,
        h_clean:     list,
        h_hack:      list,
    ) -> "AttributionCircuit":
        """
        Run offline attribution on a frozen policy + SAE.

        h_clean : list of (n_steps, n_features) arrays from clean episodes
        h_hack  : list of (n_steps, n_features) arrays from hacking episodes
        """
        print(f"Loading policy action head from {policy_path} ...")
        W_action = _load_action_head(policy_path)
        print(f"  W_action: {W_action.shape}  (n_actions x 256)")

        print(f"Loading SAE decoder from {sae_path} ...")
        W_dec = _load_sae_decoder(sae_path)
        print(f"  W_dec: {W_dec.shape}  (256 x n_features)")

        print(f"Computing attributions ({len(h_clean)} clean, {len(h_hack)} hack episodes) ...")
        data = _compute_circuit_attribution(W_action, W_dec, h_clean, h_hack)

        print(f"  Goal features ({len(data['goal_features'])}): {data['goal_features']}")
        print(f"  Hack features ({len(data['hack_features'])}): {data['hack_features']}")

        return cls(
            goal_features      = data["goal_features"],
            hack_features      = data["hack_features"],
            ie_scores          = {int(k): v for k, v in data["ie_scores"].items()},
            delta_h            = {int(k): v for k, v in data["delta_h"].items()},
            circuit_coeff_norm = {int(k): v for k, v in data["circuit_coeff_norm"].items()},
            n_clean            = data["n_clean"],
            n_hack             = data["n_hack"],
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "goal_features":      self.goal_features,
            "hack_features":      self.hack_features,
            "ie_scores":          {str(k): v for k, v in self.ie_scores.items()},
            "delta_h":            {str(k): v for k, v in self.delta_h.items()},
            "circuit_coeff_norm": {str(k): v for k, v in self.circuit_coeff_norm.items()},
            "n_clean":            self.n_clean,
            "n_hack":             self.n_hack,
        }
        json.dump(data, open(path, "w"), indent=2)
        print(f"Attribution circuit saved → {path}")

    @classmethod
    def load(cls, path: str) -> "AttributionCircuit":
        d = json.load(open(path))
        return cls(
            goal_features      = d["goal_features"],
            hack_features      = d["hack_features"],
            ie_scores          = {int(k): v for k, v in d["ie_scores"].items()},
            delta_h            = {int(k): v for k, v in d["delta_h"].items()},
            circuit_coeff_norm = {int(k): v for k, v in d["circuit_coeff_norm"].items()},
            n_clean            = d.get("n_clean", 0),
            n_hack             = d.get("n_hack",  0),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"AttributionCircuit  ({self.n_clean} clean / {self.n_hack} hack episodes)",
            "",
            f"Goal features (suppressed in hacking, top by IE):",
        ]
        for f in self.goal_features:
            lines.append(
                f"  f{f:<4d}  IE={self.ie_scores[f]:.4f}  "
                f"delta_h={self.delta_h[f]:+.4f}  "
                f"C_norm={self.circuit_coeff_norm[f]:.4f}"
            )
        lines.append(f"\nHack features (enhanced in hacking, top by IE):")
        for f in self.hack_features:
            lines.append(
                f"  f{f:<4d}  IE={self.ie_scores[f]:.4f}  "
                f"delta_h={self.delta_h[f]:+.4f}  "
                f"C_norm={self.circuit_coeff_norm[f]:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: run attribution on contrastive episode dataset
# ---------------------------------------------------------------------------

def main():
    import glob

    jsons = sorted(glob.glob(os.path.join(EP_DIR, "*.json")))
    h_clean, h_hack = [], []

    for jpath in jsons:
        meta = json.load(open(jpath))
        npz  = jpath.replace(".json", ".npz")
        if not os.path.exists(npz):
            continue
        h    = np.load(npz)["h"]
        stage   = meta.get("stage", "")
        outcome = meta.get("outcome", "")

        # Filter out episodes with out-of-distribution h values.
        # mid_induction and full_induction episodes were collected with a different
        # SAE version or before SAE calibration, producing h values up to 57M vs
        # the valid baseline range of [0, ~10].  Use only in-distribution episodes.
        if h.max() > 20.0:
            continue

        if outcome == "shortcut":
            h_hack.append(h)
        elif stage == "baseline":
            h_clean.append(h)

    print(f"Loaded {len(h_clean)} clean and {len(h_hack)} hacking episodes.")

    if not h_clean or not h_hack:
        print("ERROR: need both clean and hacking episodes.")
        return

    circuit = AttributionCircuit.discover(POLICY_PATH, SAE_PATH, h_clean, h_hack)
    print()
    print(circuit.summary())

    circuit.save(OUT_PATH)

    # Compare discovered features against known hand-labelled sets
    known_goal  = {381, 341, 119, 262, 256, 371}
    known_hack  = {195, 1, 348, 247, 111, 326, 99, 367, 327, 369, 238}
    disc_goal   = set(circuit.goal_features)
    disc_hack   = set(circuit.hack_features)

    print(f"\nOverlap with hand-labelled sets:")
    print(f"  Goal features  overlap: {len(disc_goal & known_goal)}/{len(known_goal)}"
          f"  ({sorted(disc_goal & known_goal)})")
    print(f"  Hack features  overlap: {len(disc_hack & known_hack)}/{len(known_hack)}"
          f"  ({sorted(disc_hack & known_hack)})")
    print(f"  New features not in known sets: "
          f"{sorted((disc_goal | disc_hack) - (known_goal | known_hack))}")


if __name__ == "__main__":
    main()
