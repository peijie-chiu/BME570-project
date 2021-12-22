import numpy as np
import matplotlib.pyplot as plt

###################### Plot Dice and Signal-Area ######################

with open("../data/raw/data.npy", "rb") as f:
    tsa4 = np.load(f)[0, ..., -100:].sum(axis=(0, 1)) / 64 / 64

print(tsa4.shape)

with open("../cache/exps/32x32/Exp_bsz16_eps200/dices.npy", "rb") as f:
    dice1 = np.load(f)
with open("../cache/exps/32x32/Exp_bsz16_eps200/tsas.npy", "rb") as f:
    tsa1 = np.load(f) / 32 / 32
    
with open("../cache/exps/16x16/Exp_bsz16_eps200/dices.npy", "rb") as f:
    dice2 = np.load(f)
with open("../cache/exps/16x16/Exp_bsz16_eps200/tsas.npy", "rb") as f:
    tsa2 = np.load(f) / 16 / 16

with open("../cache/exps/8x8/Exp_bsz16_eps200/dices.npy", "rb") as f:
    dice3 = np.load(f)
with open("../cache/exps/8x8/Exp_bsz16_eps200/tsas.npy", "rb") as f:
    tsa3 = np.load(f) / 8 / 8

with open("../data/raw/sigma.npy", "rb") as f:
    sigma_s = np.load(f)

test_sigma_s = sigma_s[-100:]
idx = np.argsort(test_sigma_s)

plt.figure()
plt.suptitle("DSC")
plt.boxplot([dice1, dice2,dice3], boxprops=dict(facecolor='b'), patch_artist=True)
plt.title("Dice Similarity Coefficients")
plt.xticks([1, 2, 3], ['32x32', '16x16', '8x8'])
plt.xlabel("Number of Pixels")
plt.ylabel("Dice Similarity Coefficients")
plt.savefig("../cache/stats.jpg")

test_sigma_s = np.around(test_sigma_s[idx], 4)
plt.figure(figsize=(10, 8))
plt.suptitle("True Signal Area (Prediction Area v.s. Truth Area)")
plt.subplot(4, 1, 1)
plt.title("True Signal Area")
plt.plot(tsa4[idx], label="Truth")
plt.xticks([], [])

plt.subplot(4, 1, 2)
plt.title("Predicted Signal Area 32x32")
plt.plot(tsa1[:, 0][idx], label="Predicted")
plt.xticks([], [])

plt.subplot(4, 1, 3)
plt.title("Predicted Signal Area 16x16")
plt.plot(tsa2[:, 0][idx], label="Predicted")
plt.xticks([], [])
plt.ylabel("Signal Area")

plt.subplot(4, 1, 4)
plt.title("Predicted Signal Area 8x8")
plt.plot(tsa3[:, 0][idx], label="Predicted")
plt.xticks(np.arange(0, 100, 20), test_sigma_s[::20])
plt.xlabel("Sigma")

plt.savefig("../cache/true_signal_area.jpg")

# ###################### Visualize loss ######################
# with open("../cache/exps/32x32/Exp_bsz16_eps200/logs.npy", "rb") as f:
#     loss1 = np.load(f)
#     dices1 = np.load(f)

# with open("../cache/exps/16x16/Exp_bsz16_eps200/logs.npy", "rb") as f:
#     loss2 = np.load(f)
#     dices2 = np.load(f)

# with open("../cache/exps/8x8/Exp_bsz16_eps200/logs.npy", "rb") as f:
#     loss3 = np.load(f)
#     dices3 = np.load(f)

# train_dices1, val_dices1 = dices1[:, 0], dices1[:, 1]
# train_dices2, val_dices2 = dices2[:, 0], dices2[:, 1]
# train_dices3, val_dices3 = dices3[:, 0], dices3[:, 1]
# train_loss1, val_loss1 = loss1[:, 0], loss1[:, 1]
# train_loss2, val_loss2 = loss2[:, 0], loss2[:, 1]
# train_loss3, val_loss3 = loss3[:, 0], loss3[:, 1]

# plt.figure()
# plt.title("Dice Similarity Coeeficients (Training/Validation)")
# plt.plot(train_dices1, label="32x32 training")
# plt.plot(val_dices1, label="32x32 validation")
# plt.plot(train_dices2, label="16x16 training")
# plt.plot(val_dices2, label="16x16 validation")
# plt.plot(train_dices3, label="8x8 training")
# plt.plot(val_dices3, label="8x8 validation")
# plt.xlabel("Epoch")
# plt.ylabel("DSC")
# plt.legend()
# plt.savefig("../cache/dice.jpg")

# plt.figure()
# plt.title("Dice Loss (Training/Validation)")
# plt.plot(train_loss1, label="32x32 training")
# plt.plot(val_loss1, label="32x32 validation")
# plt.plot(train_loss2, label="16x16 training")
# plt.plot(val_loss2, label="16x16 validation")
# plt.plot(train_loss3, label="8x8 training")
# plt.plot(val_loss3, label="8x8 validation")
# plt.xlabel("Epoch")
# plt.ylabel("Dice Loss")
# plt.legend()
# plt.savefig("../cache/loss.jpg")