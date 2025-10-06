import torch
from torch.utils.data import DataLoader
from models import PilotNet, PilotNetSwish, ResNetPilot, VGGPilot
from dataset import PilotNetDataset
import numpy as np

def evaluate_model(model_class, weight_path, test_csv, root="dataset", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    ds = PilotNetDataset(root, test_csv, max_angle_rad=0.6, crop=(20,8,0,0), resize=(66,200), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.HuberLoss(delta=1.0)
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            total_loss += loss_fn(y_pred, y).item() * x.size(0)
            preds.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    total_loss /= len(loader.dataset)
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    print(f"{model_class.__name__}: mean test loss = {total_loss:.5f}")
    return preds, targets


if __name__ == "__main__":
    # Example test dataset
    test_csv = "dataset/rain_test_20251001/labels.csv"

    models_to_test = [
        (PilotNet, "pilotnet_best.pt"),
        (PilotNetSwish, "pilotnet_swish_best.pt"),
        (ResNetPilot, "resnetpilot_best.pt"),
        (VGGPilot, "vggpilot_best.pt"),
    ]

    for model_class, weight_path in models_to_test:
        evaluate_model(model_class, weight_path, test_csv)
