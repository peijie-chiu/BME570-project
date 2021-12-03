import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import SegDataset
from model import Unet
from metric import WeightedDiceLoss, DiceCoeff

torch.manual_seed(0)


def eval_epoch(test_dl, model, criterion, eval_metric):
    with torch.no_grad():
        epoch_loss = 0.
        epoch_dice = 0.
        model.eval()
        for idx, (x, y) in enumerate(test_dl):
            x, y = x.to(device), y.to(device)
            bsz = x.size(0)

            logits = model(x)
            probs = torch.sigmoid(logits)
            loss = criterion(y, probs) 
            
            epoch_loss += loss.item() * bsz
            y_pred = probs >= 0.5
            epoch_dice += eval_metric(y, y_pred).item() * bsz

    return epoch_loss / len(test_dl.dataset), epoch_dice / len(test_dl.dataset)


def train_epoch(train_dl, model, optimizer, criterion, eval_metric):
    epoch_loss = 0.
    epoch_dice = 0.
    running_num = 0
    model.train()
    for n_iter, (x, y) in enumerate(train_dl):
        bsz = x.size(0)
        running_num += bsz

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        probs = torch.sigmoid(logits)
        loss = criterion(y, probs) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bsz
        y_pred = probs >= 0.5
        epoch_dice += eval_metric(y, y_pred).item() * bsz

        if (n_iter + 1) % dispatcher == 0:
            print(f"[{running_num}/{len(train_dl.dataset)}] training loss:{epoch_loss / running_num:.4f} training dice: {epoch_dice / running_num:.4f}")

    return epoch_loss / len(train_dl.dataset), epoch_dice / len(train_dl.dataset)


############################## Training ##############################
input_dir = os.path.join('cache', 'inputs/recon_32')
save_dir = os.path.join('cache', 'exps/recon_32')

try:
    os.mkdir(save_dir)
except:
    pass

batch_size = 8
epochs = 100
lr = 5e-4
dispatcher = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dir = os.path.join(input_dir, 'train_set.pt')
val_dir = os.path.join(input_dir, 'val_set.pt')
train_set = torch.load(train_dir)
val_set = torch.load(val_dir)

train_dl = DataLoader(train_set, batch_size=batch_size)
val_dl = DataLoader(val_set, batch_size=batch_size)

model = Unet(in_channels=1, n_class=1).to(device)
criterion = WeightedDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-4)

eval_metric = DiceCoeff()

save_path = os.path.join(save_dir, f"Exp_bsz{batch_size}_eps{epochs}")

try:
    os.mkdir(save_path)
except:
    pass


############################## Training Loop ##############################
losses = list()
dices = list()

for ep in range(epochs):
    train_loss, train_dice = train_epoch(train_dl, model, optimizer, criterion, eval_metric)
    val_loss, val_dice = eval_epoch(val_dl, model, criterion, eval_metric)
    
    print(f"{'='*10} Epoch:[{ep+1}/{epochs}] val loss: {val_loss:.4f} val dice: {val_dice:.4f} {'='*10}")
    losses.append([train_loss, val_loss])
    dices.append([train_dice, val_dice])

    torch.save(model.state_dict(), os.path.join(save_path, f"ep{ep}.pt"))

    scheduler.step(val_loss)

losses = np.asarray(losses)
dices = np.asarray(dices)

with open(os.path.join(save_path, "logs.npy"), "wb") as f:
    np.save(f, losses)
    np.save(f, dices)

