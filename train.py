import os
import torch
from torch.utils.data import DataLoader, random_split
from models import PilotNet, PilotNetSwish, ResNetPilot, VGGPilot
from dataset import PilotNetDataset
import numpy as np
import matplotlib.pyplot as plt


# -------------------- TRAINING FUNCTION --------------------
def train_and_save(model_class, name, train_loader, val_loader, device, epochs=10, lr=1e-3):
    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval(); val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{name}] Epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), f"{name}_best.pt")


# -------------------- EVALUATION FUNCTION --------------------
def evaluate_model(model_class, weight_path, test_csv, root="dataset", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    ds = PilotNetDataset(root, test_csv,
                         max_angle_rad=0.6,
                         crop=(20,8,0,0),
                         resize=(66,200),
                         augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    mae = np.mean(np.abs(preds - targets))
    return mae


# -------------------- MAIN --------------------
def main():
    root = "dataset"
    csv_path = "dataset/clear_20250921_194106/labels.csv"  # your clear training dataset
    batch_size, epochs, lr = 128, 1, 1e-3
    max_angle_rad = 0.6

    # Dataset loading and splitting
    ds = PilotNetDataset(root, csv_path,
                         max_angle_rad=max_angle_rad,
                         crop=(20,8,0,0),
                         resize=(66,200),
                         augment=True)
    n_val = max(1000, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models to train
    models = [
        (PilotNet, "pilotnet"),
        (PilotNetSwish, "pilotnet_swish"),
        (ResNetPilot, "resnet_pilot"),
        (VGGPilot, "vgg_pilot")
    ]

    # 1️⃣ Train and save all models
    for model_class, name in models:
        train_and_save(model_class, name, train_loader, val_loader, device, epochs, lr)

    # 2️⃣ Automatically find all test datasets (clear, fog, rain, etc.)
    test_dirs = [d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))
                 and "test" in d.lower()]

    print(f"\nFound test datasets: {test_dirs}\n")

    # 3️⃣ Evaluate each model on each weather test set
    results = {m.__name__: {} for m, _ in models}

    for test_dir in test_dirs:
        test_csv = os.path.join(root, test_dir, "labels.csv")
        weather = test_dir.split("_")[0]  # e.g. "clear", "fog", "rain"
        for model_class, name in models:
            weight_path = f"{name}_best.pt"
            mae = evaluate_model(model_class, weight_path, test_csv)
            results[model_class.__name__][weather] = mae
            print(f"{model_class.__name__} on {weather}: MAE = {mae:.5f}")

    # 4️⃣ Print summary
    print("\n=== Summary (MAE in radians) ===")
    for model, weathers in results.items():
        for weather, mae in weathers.items():
            print(f"{model:15s} | {weather:8s} | MAE = {mae:.5f}")

    # 5️⃣ Plot grouped bar chart
    weathers = sorted({w for vals in results.values() for w in vals})
    models_names = list(results.keys())
    x = np.arange(len(weathers))
    width = 0.18

    plt.figure(figsize=(8, 5))
    for i, model in enumerate(models_names):
        maes = [results[model].get(w, np.nan) for w in weathers]
        plt.bar(x + i * width, maes, width=width, label=model)

    plt.xticks(x + width * (len(models_names) - 1) / 2, weathers)
    plt.ylabel("Mean Absolute Error (radians)")
    plt.title("Open-loop MAE across Weather Conditions and Models")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
