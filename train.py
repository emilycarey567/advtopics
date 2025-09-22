import torch
from torch.utils.data import DataLoader, random_split
from pilotnet import PilotNet
from dataset import PilotNetDataset

def main():
    root = "dataset"
    csv_path = "dataset/clear_20250921_194106/labels.csv"  # update as needed
    max_angle_rad = 0.6
    batch_size, epochs, lr = 128, 10, 1e-3

    ds = PilotNetDataset(root, csv_path,
                         max_angle_rad=max_angle_rad,
                         crop=(20,8,0,0),
                         resize=(66,200),
                         augment=True)

    n_val = max(1000, int(0.1*len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)  
    val_loader   = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    best = float("inf")
    for epoch in range(1, epochs+1):
        model.train(); train_loss=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()*x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval(); val_loss=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()*x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "pilotnet_best.pt")

if __name__ == "__main__":
    main()
