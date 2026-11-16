import copy, random
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = "data/plantvillage"
IMG = 224
PER_CLASS = 300
TEST_FRAC = 0.2
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((IMG, IMG)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)),
                              transforms.ToTensor(), norm])

def load_splits():
    base = datasets.ImageFolder(DATA_DIR)
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for i, (_, y) in enumerate(base.samples):
        by_class[y].append(i)
    train_idx, test_idx = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        idxs = idxs[:PER_CLASS]
        n_test = int(TEST_FRAC * len(idxs))
        test_idx += idxs[:n_test]
        train_idx += idxs[n_test:]
    return base, base.classes, train_idx, test_idx

def make_model(n_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m.to(device)

def local_train(global_model, loader, epochs, lr):
    model = copy.deepcopy(global_model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    preds, truths = [], []
    for x, y in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        truths += y.tolist()
    return preds, truths