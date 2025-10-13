import torch
from torch.utils.data import DataLoader, random_split
from models import PilotNet, PilotNetSwish, ResNetPilot, VGGPilot
from dataset import PilotNetDataset
import matplotlib.pyplot as plt
import os
from torch.cuda.amp import autocast, GradScaler

# ============================================================
# Global speed settings
# ============================================================
torch.backends.cudnn.benchmark = True  # optimise conv kernels for fixed input size

# ============================================================
# Plotting helper
# ============================================================
def plot_loss_curves(train_curve, val_curve, name):
    plt.figure()
    plt.plot(train_curve, label="Training Loss")
    plt.plot(val_curve, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title(f"Training vs Validation Loss — {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}_loss_curve.png")
    plt.close()
    print(f"[{name}] 📉 Saved loss curve to plots/{name}_loss_curve.png")


# ============================================================
# Training function with mixed precision + early stopping
# ============================================================
def train_and_save(model_class, name, train_loader, val_loader, device,
                   epochs=50, lr=1e-3, patience=5, delta=0.001):
    print(f"\n🚀 Training {name} on device: {device}")
    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    scaler = GradScaler()  # for AMP

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    train_curve, val_curve = [], []

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():                     # mixed precision forward pass
                preds = model(x)
                loss = loss_fn(preds, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast():
                    val_loss += loss_fn(model(x), y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        train_curve.append(train_loss)
        val_curve.append(val_loss)

        print(f"[{name}] Epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")

        # ---- Early stopping ----
        if best_val - val_loss > delta:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), f"{name}_best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"[{name}] ⏹ Early stopping at epoch {epoch} (best @ {best_epoch})")
                break

    # ---- Save plot ----
    plot_loss_curves(train_curve, val_curve, name)
    print(f"[{name}] ✅ Finished. Best val loss = {best_val:.5f} @ epoch {best_epoch}")
    return train_curve, val_curve


# ============================================================
# Main
# ============================================================
def main():
    root = "dataset"
    csv_path = "dataset/clear_20250921_194106/labels.csv"  # update as needed
    batch_size, epochs, lr = 256, 5, 1e-3
    max_angle_rad = 0.6

    ds = PilotNetDataset(root, csv_path,
                         max_angle_rad=max_angle_rad,
                         crop=(20, 8, 0, 0),
                         resize=(66, 200),
                         augment=True)
    n_val = max(1000, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB\n")

    models = [
        (ResNetPilot, "resnet_pilot"),
        (VGGPilot, "vgg_pilot"),
        (PilotNet, "pilotnet"),
        (PilotNetSwish, "pilotnet_swish")
    ]

    for model_class, name in models:
        train_and_save(model_class, name, train_loader, val_loader, device, epochs, lr)


if __name__ == "__main__":
    main()
