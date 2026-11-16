import sys, json, os
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score
from torchvision import datasets
from common import *

mode = sys.argv[1]  # "iid" or "noniid"
N_FARMS, ROUNDS, LOCAL_EPOCHS, LR, BS, ALPHA = 3, 5, 1, 1e-4, 32, 0.5
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

base, classes, train_idx, test_idx = load_splits()
labels = np.array([y for _, y in base.samples])
train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
test_ds = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=64)

def split_clients(idx, mode):
    idx = np.array(idx)
    rng = np.random.default_rng(SEED)
    if mode == "iid":
        rng.shuffle(idx)
        return [np.array(p) for p in np.array_split(idx, N_FARMS)]
    clients = [[] for _ in range(N_FARMS)]
    for c in range(len(classes)):
        cidx = idx[labels[idx] == c]
        rng.shuffle(cidx)
        p = rng.dirichlet([ALPHA] * N_FARMS)
        cuts = (np.cumsum(p) * len(cidx)).astype(int)[:-1]
        for k, part in enumerate(np.split(cidx, cuts)):
            clients[k].extend(part.tolist())
    return [np.array(c) for c in clients]

def fedavg(states, sizes):
    total = sum(sizes)
    avg = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            avg[k] = sum(s[k] * (n / total) for s, n in zip(states, sizes))
        else:
            avg[k] = states[0][k].clone()
    return avg

clients = split_clients(train_idx, mode)
print("Farm sizes:", [len(c) for c in clients])
for k, c in enumerate(clients):
    print(f"Farm {k+1} class counts:", np.bincount(labels[c], minlength=len(classes)).tolist())

global_model = make_model(len(classes))
size_mb = sum(p.numel() * p.element_size() for p in global_model.parameters()) / 1e6
log = []
for r in range(1, ROUNDS + 1):
    states, sizes = [], []
    for farm in clients:
        loader = DataLoader(Subset(train_ds, farm.tolist()), batch_size=BS, shuffle=True)
        states.append(local_train(global_model, loader, LOCAL_EPOCHS, LR))
        sizes.append(len(farm))
    global_model.load_state_dict(fedavg(states, sizes))
    preds, truths = predict_all(global_model, test_loader)
    acc = accuracy_score(truths, preds)
    mb_sent = 2 * N_FARMS * size_mb
    log.append({"round": r, "test_accuracy": acc, "mb_sent_this_round": round(mb_sent, 1)})
    print(f"Round {r}/{ROUNDS}  test_acc={acc:.4f}  MB sent={mb_sent:.1f}")

report = classification_report(truths, preds, target_names=classes, output_dict=True, zero_division=0)
torch.save(global_model.state_dict(), f"model/fedavg_{mode}.pt")
json.dump({"method": f"fedavg_{mode}", "rounds": log,
           "final_test_accuracy": log[-1]["test_accuracy"],
           "total_mb_sent": round(sum(x["mb_sent_this_round"] for x in log), 1),
           "report": report}, open(f"results/fedavg_{mode}.json", "w"), indent=2)
print(f"Saved model/fedavg_{mode}.pt and results/fedavg_{mode}.json")