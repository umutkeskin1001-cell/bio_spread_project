"""Quick diagnostic: cold-start vs temporal AUC on synthetic data."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bio_spread.models.sovereign import BioSpreadModel
from bio_spread.models.trainer import BioSpreadTrainer


class ColdStartDiagnosticDataset:
    def __init__(self, B=200, L=10, n_static=12, n_snapshot=15):
        self.B = B
        self.L = L
        self.n_static = n_static
        self.n_snapshot = n_snapshot
        torch.manual_seed(42)
        self.static = torch.randn(B, n_static)
        self.seq = torch.randn(B, L, n_snapshot)
        self.mask = torch.ones(B, L)
        self.hazard = torch.zeros(B, L, 3)
        for h in range(3):
            base = torch.sigmoid(self.static[:, 0] * 0.5 + torch.randn(B) * 0.3)
            self.hazard[:, :, h] = base.unsqueeze(1).expand(-1, L)
            noise = torch.rand(B, L) * 0.2
            self.hazard[:, :, h] = (self.hazard[:, :, h] + noise).clamp(0, 1).round()
        self.counts = torch.poisson(torch.ones(B) * 2.0)
        self.seq_len = torch.full((B,), L)

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        return {
            "static": self.static[idx],
            "seq": self.seq[idx],
            "mask": self.mask[idx],
            "hazard": self.hazard[idx],
            "count": self.counts[idx],
            "seq_len": self.seq_len[idx],
            "backbone_id": f"syn_{idx}",
        }


def syn_collate(batch, max_seq_len):
    from bio_spread.data.dataset import sequence_collate
    return sequence_collate(batch, max_seq_len)


def evaluate_cold_vs_temporal(trainer, loader, device):
    trainer.model.eval()
    all_cold_probs = {h: [] for h in range(3)}
    all_temporal_probs = {h: [] for h in range(3)}
    all_targets = {h: [] for h in range(3)}

    from sklearn.metrics import roc_auc_score

    with torch.no_grad():
        for batch in loader:
            static = batch["static"].to(device)
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["seq_len"].to(device)
            targets = batch["hazard"].to(device)

            temporal_mask = torch.ones(static.size(0), dtype=torch.bool, device=device)
            out_cold = trainer.model(static, seq, mask, temporal_mask=temporal_mask)

            temporal_mask_false = torch.zeros(static.size(0), dtype=torch.bool, device=device)
            out_temporal = trainer.model(static, seq, mask, temporal_mask=temporal_mask_false)

            idx = (lengths - 1).clamp(min=0)
            last_targets = targets[range(static.size(0)), idx]

            cold_probs = torch.sigmoid(out_cold.hazard_logits)
            temporal_probs = torch.sigmoid(out_temporal.hazard_logits)

            for h in range(3):
                valid = last_targets[:, h] >= 0
                if valid.any():
                    all_cold_probs[h].append(cold_probs[valid, h].cpu().numpy())
                    all_temporal_probs[h].append(temporal_probs[valid, h].cpu().numpy())
                    all_targets[h].append(last_targets[valid, h].cpu().numpy())

    cold_aucs = []
    temporal_aucs = []
    for h in range(3):
        if all_cold_probs[h]:
            cold_p = torch.cat([torch.tensor(p) for p in all_cold_probs[h]]).numpy()
            temporal_p = torch.cat([torch.tensor(p) for p in all_temporal_probs[h]]).numpy()
            t = torch.cat([torch.tensor(p) for p in all_targets[h]]).numpy()
            if len(set(t)) > 1:
                cold_auc = roc_auc_score(t, cold_p)
                temporal_auc = roc_auc_score(t, temporal_p)
                cold_aucs.append(cold_auc)
                temporal_aucs.append(temporal_auc)
                print(f"  H{h+1}: Cold AUC={cold_auc:.4f}, Temporal AUC={temporal_auc:.4f}")
            else:
                print(f"  H{h+1}: Single class (pos_rate={t.mean():.3f}), skipping AUC")

    avg_cold = sum(cold_aucs) / len(cold_aucs) if cold_aucs else 0.0
    avg_temporal = sum(temporal_aucs) / len(temporal_aucs) if temporal_aucs else 0.0
    return avg_cold, avg_temporal


def main():
    print("=" * 60)
    print("COLD-START DIAGNOSTIC")
    print("=" * 60)

    B, L, n_static, n_snapshot = 200, 10, 12, 15
    ds = ColdStartDiagnosticDataset(B=B, L=L, n_static=n_static, n_snapshot=n_snapshot)
    loader = DataLoader(ds, batch_size=32, collate_fn=lambda b: syn_collate(b, L))

    device = "cpu"

    print("\n--- Configuration 1: All flags ON (evidential + retrieval + cagrad) ---")
    torch.manual_seed(42)
    model1 = BioSpreadModel(
        n_static=n_static, n_snapshot=n_snapshot,
        static_dim=64, temporal_dim=64, hidden_dim=64, n_hazard=3,
        use_evidential=True, use_retrieval=True,
        prototype_dim=64, prototype_k=5,
    )
    trainer1 = BioSpreadTrainer(
        model1, device=device, epochs=15, patience=20, warmup_epochs=2,
        lambda_cold=1.0, lambda_kd=1.0, use_cagrad=True,
        temporal_masking_prob=0.3, use_curriculum=True,
        calibrate=False,
    )
    trainer1.fit(loader, val_loader=loader)
    cold_auc1, temporal_auc1 = evaluate_cold_vs_temporal(trainer1, loader, device)
    print(f"  AVG: Cold AUC={cold_auc1:.4f}, Temporal AUC={temporal_auc1:.4f}")

    print("\n--- Configuration 2: Standard (no evidential, no retrieval) ---")
    torch.manual_seed(42)
    model2 = BioSpreadModel(
        n_static=n_static, n_snapshot=n_snapshot,
        static_dim=64, temporal_dim=64, hidden_dim=64, n_hazard=3,
        use_evidential=False, use_retrieval=False,
    )
    trainer2 = BioSpreadTrainer(
        model2, device=device, epochs=15, patience=20, warmup_epochs=2,
        lambda_cold=1.0, lambda_kd=1.0, use_cagrad=False,
        temporal_masking_prob=0.3, use_curriculum=True,
        calibrate=False,
    )
    trainer2.fit(loader, val_loader=loader)
    cold_auc2, temporal_auc2 = evaluate_cold_vs_temporal(trainer2, loader, device)
    print(f"  AVG: Cold AUC={cold_auc2:.4f}, Temporal AUC={temporal_auc2:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Config 1 (Full): Cold={cold_auc1:.4f}, Temporal={temporal_auc1:.4f}, Gap={temporal_auc1-cold_auc1:.4f}")
    print(f"  Config 2 (Std):  Cold={cold_auc2:.4f}, Temporal={temporal_auc2:.4f}, Gap={temporal_auc2-cold_auc2:.4f}")

    if cold_auc1 > 0.75:
        print("\n  Cold-start AUC > 0.75: EXCELLENT")
    elif cold_auc1 > 0.70:
        print("\n  Cold-start AUC > 0.70: GOOD")
    elif cold_auc1 > 0.60:
        print("\n  Cold-start AUC > 0.60: ACCEPTABLE")
    else:
        print("\n  Cold-start AUC < 0.60: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
