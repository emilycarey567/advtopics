import torch
from torch.utils.data import DataLoader
from models import PilotNet, PilotNetSwish, ResNetPilot, VGGPilot
from dataset import PilotNetDataset
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model_class, weight_path, test_csv, root="dataset", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model_class().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()


    ds = PilotNetDataset(root, test_csv,
                         max_angle_rad=0.6,
                         crop=(20, 8, 0, 0),
                         resize=(66, 200),
                         augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)


    huber_fn = torch.nn.HuberLoss(delta=1.0)
    total_huber = 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

      
            total_huber += huber_fn(y_pred, y).item() * x.size(0)

            
            preds.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())


    total_huber /= len(loader.dataset)
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    mae = np.mean(np.abs(preds - targets))

    print(f"{model_class.__name__}: Huber = {total_huber:.5f}, MAE = {mae:.5f} radians")
    return mae, total_huber


if __name__ == "__main__":

    test_csv = "dataset/test_fog_20251001/labels.csv"

    models_to_test = [
        (PilotNet, "pilotnet_best.pt"),
        (PilotNetSwish, "pilotnet_swish_best.pt"),
        (ResNetPilot, "resnet_pilot_best.pt"),
        (VGGPilot, "vgg_pilot_best.pt"),
    ]


    mae_results = {}
    for model_class, weight_path in models_to_test:
        mae, _ = evaluate_model(model_class, weight_path, test_csv)
        mae_results[model_class.__name__] = mae

    # Plot MAE results 
    names = list(mae_results.keys())
    values = list(mae_results.values())

    plt.figure(figsize=(7, 5))
    plt.bar(names, values, color='skyblue')
    plt.ylabel("Mean Absolute Error (radians)")
    plt.title("Open-loop Evaluation: MAE across CNN Architectures")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
