#!/usr/bin/env python3
"""
train_grokking.py — generate grokking training curves across
multiple seeds and weight decay values.

trains a small transformer on modular addition (mod 97).
saves epoch-by-epoch train/test accuracy to CSV.

designed to run on GPU (boron: 2x Titan RTX).
each run takes ~5-10 min. full sweep: ~1 hour.

usage:
    python train_grokking.py                    # default: 10 seeds × 5 WD values
    python train_grokking.py --seeds 5 --gpu 0  # 5 seeds on GPU 0
"""

import argparse
import os
import csv
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# modular arithmetic dataset
# ─────────────────────────────────────────────

def make_mod_data(p=97, op="add", train_frac=0.5):
    """
    create modular arithmetic dataset.
    x op y = z (mod p), where op is addition.
    split into train/test by random selection of (x,y) pairs.
    """
    pairs = [(x, y) for x in range(p) for y in range(p)]
    if op == "add":
        labels = [(x + y) % p for x, y in pairs]
    elif op == "subtract":
        labels = [(x - y) % p for x, y in pairs]
    elif op == "multiply":
        labels = [(x * y) % p for x, y in pairs]
    else:
        raise ValueError(f"unknown op: {op}")

    # deterministic split based on index
    n = len(pairs)
    n_train = int(n * train_frac)

    # shuffle deterministically
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=gen)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_all = torch.tensor(pairs)  # (n, 2)
    y_all = torch.tensor(labels)  # (n,)

    return (x_all[train_idx], y_all[train_idx],
            x_all[test_idx], y_all[test_idx])


# ─────────────────────────────────────────────
# transformer model (small, following Power et al.)
# ─────────────────────────────────────────────

class GrokTransformer(nn.Module):
    """
    small transformer for modular arithmetic.
    2 tokens in, 1 classification out.
    """
    def __init__(self, p=97, d_model=128, n_heads=4, n_layers=2, dropout=0.0):
        super().__init__()
        self.p = p
        self.embed = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # 2 input positions + 1 output

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, p)

    def forward(self, x):
        # x: (batch, 2) — two input tokens
        batch_size = x.shape[0]
        tok_emb = self.embed(x)  # (batch, 2, d_model)
        pos = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(pos)
        h = tok_emb + pos_emb
        h = self.transformer(h)
        # use mean of both positions for classification
        h = h.mean(dim=1)  # (batch, d_model)
        return self.head(h)  # (batch, p)


# ─────────────────────────────────────────────
# training loop
# ─────────────────────────────────────────────

def train_one_run(seed, weight_decay, p=97, n_epochs=7500,
                  lr=1e-3, batch_size=512, device="cuda",
                  log_every=10):
    """
    train one grokking run and return epoch-by-epoch metrics.
    """
    torch.manual_seed(seed)

    x_train, y_train, x_test, y_test = make_mod_data(p=p)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
    )

    model = GrokTransformer(p=p).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    metrics = []
    t0 = time.time()

    for epoch in range(n_epochs):
        # train
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0

        for xb, yb in train_loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_correct += (logits.argmax(-1) == yb).sum().item()
            train_total += len(yb)
            train_loss_sum += loss.item() * len(yb)

        # eval (every log_every epochs)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
                test_loss = F.cross_entropy(test_logits, y_test).item()

            train_acc = train_correct / train_total
            train_loss = train_loss_sum / train_total

            metrics.append({
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
            })

            # early stop if fully grokked
            if test_acc > 0.99 and epoch > 500:
                # log a few more to confirm persistence
                for extra in range(1, 6):
                    e = epoch + extra * log_every
                    if e < n_epochs:
                        # quick eval
                        for xb, yb in train_loader:
                            logits = model(xb)
                            loss = F.cross_entropy(logits, yb)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        with torch.no_grad():
                            test_logits = model(x_test)
                            ta = (test_logits.argmax(-1) == y_test).float().mean().item()
                        metrics.append({
                            "epoch": e,
                            "train_acc": train_acc,
                            "test_acc": ta,
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                        })
                break

    elapsed = time.time() - t0
    final_test = metrics[-1]["test_acc"]
    grokked = final_test > 0.95

    return {
        "seed": seed,
        "weight_decay": weight_decay,
        "n_epochs_run": metrics[-1]["epoch"],
        "final_test_acc": final_test,
        "grokked": grokked,
        "elapsed_s": elapsed,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────
# main: sweep over seeds × weight decay
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=7500)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    # weight decay values: from no-grok to strong-grok
    wd_values = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]

    all_results = []

    for wd in wd_values:
        for seed in range(args.seeds):
            print(f"\n--- WD={wd}, seed={seed} ---")
            result = train_one_run(
                seed=seed, weight_decay=wd,
                n_epochs=args.epochs, device=device,
            )
            all_results.append(result)

            status = "GROKKED" if result["grokked"] else "no grok"
            print(f"  {status}: test_acc={result['final_test_acc']:.4f}, "
                  f"{result['elapsed_s']:.0f}s, {result['n_epochs_run']} epochs")

            # save per-run CSV
            csv_path = os.path.join(
                args.output_dir,
                f"run_wd{wd}_seed{seed}.csv"
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "epoch", "train_acc", "test_acc", "train_loss", "test_loss"
                ])
                writer.writeheader()
                writer.writerows(result["metrics"])

    # save summary
    summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "weight_decay", "grokked", "final_test_acc",
            "n_epochs_run", "elapsed_s"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items() if k != "metrics"})

    # print summary table
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'WD':>6s} | {'grokked':>8s} | {'rate':>6s}")
    print("-" * 30)
    for wd in wd_values:
        runs = [r for r in all_results if r["weight_decay"] == wd]
        n_grok = sum(1 for r in runs if r["grokked"])
        print(f"{wd:6.2f} | {n_grok:>3d}/{len(runs):<3d} | {n_grok/len(runs):>5.0%}")

    print(f"\ntotal runs: {len(all_results)}")
    print(f"output: {args.output_dir}/")


if __name__ == "__main__":
    main()
