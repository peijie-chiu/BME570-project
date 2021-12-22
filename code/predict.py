import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader import SegDataset
from model import Unet
from metric import DiceCoeff
from utils import save_figure
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_best_weights(path, save_best_model_only=True):
    with open(os.path.join(path, "logs.npy"), "rb") as f:
        losses = np.load(f)
        dices = np.load(f)

    train_losses, val_losses = losses[:, 0], losses[:, 1]
    train_dices, val_dices = dices[:, 0], dices[:, 1]

    plt.figure()
    plt.title("losses")
    plt.plot(train_losses, label="training loss")
    plt.plot(val_losses, label="validation loss")
    plt.legend()
    plt.savefig(os.path.join(path, "losses.jpg"))
    plt.close()

    plt.figure()
    plt.title("dices")
    plt.plot(train_dices, label="training dice")
    plt.plot(val_dices, label="validation dice")
    plt.legend()
    plt.savefig(os.path.join(path, "dice.jpg"))
    plt.close()

    if os.path.exists(os.path.join(path, "best_model.pt")):
        return torch.load(os.path.join(path, "best_model.pt"))

    best_model_idx = np.argmax(val_dices)
    best_model_path = os.path.join(path, f"ep{best_model_idx}.pt")
    best_model = torch.load(best_model_path)

    if save_best_model_only:
        os.rename(best_model_path, os.path.join(path, "best_model.pt"))  
        remove_path = glob(os.path.join(path, "ep*.pt"))
        for p in remove_path:
            os.remove(p)

    return best_model


def predict_img(test_dl, model, eval_metric, save_dir):
    save_path = os.path.join(save_dir, "prediction")
    titles = ['Input', 'Predict', 'Truth']
    dices = list()
    tsas = list()
    try:
        os.mkdir(save_path)
    except:
        pass

    with torch.no_grad():
        model.eval()
        num = 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            bsz = x.size(0)

            for idx in range(bsz):
                image = x[None, idx, ...]
                y_true = y[None, idx, ...]
                logits = model(image)
                probs = torch.sigmoid(logits)
                y_pred = probs >= 0.5
                dice = eval_metric(y_true, y_pred).item()
                dices.append(dice)

                y_pred = torch.squeeze(y_pred).detach().cpu().numpy()
                y_true = torch.squeeze(y_true).detach().cpu().numpy()
                image = torch.squeeze(image).detach().cpu().numpy()
                
                tsas.append([y_pred.sum(), y_true.sum()])

                save_figure([image, y_pred, y_true], titles, os.path.join(save_path, f"{num}.jpg"), None)
                num += 1
    # print(np.mean(dices), np.mean(tsas))
    with open(f"{save_dir}/dices.npy", "wb") as f:
        np.save(f, np.array(dices))

    with open(f"{save_dir}/tsas.npy", "wb") as f:
        np.save(f, np.array(tsas))

    # Boxplot of Dice Score
    plt.figure()
    plt.title("Boxplot of Dice Score")
    plt.boxplot(dices)
    plt.savefig(os.path.join(save_dir, "boxplot.jpg"))
    plt.close()

pixels = 8
input_dir = os.path.join('../data', f'{pixels}x{pixels}')
save_dir = os.path.join('../cache', f'exps/{pixels}x{pixels}')

# get the best model
weights = "Exp_bsz16_eps200"
net = get_best_weights(os.path.join(save_dir, weights))

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_set = SegDataset(os.path.join(input_dir, "test_set.npy"))
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Unet(in_channels=1, n_class=1, layers=3, n_base_filters=32).to(device)
model.load_state_dict(net)
eval_metric = DiceCoeff()

predict_img(test_dl, model, eval_metric, os.path.join(save_dir, weights))