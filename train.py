import json, os, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

DATA_DIR = "data/plantvillage"
MODEL_DIR = "model"
EPOCHS = 5
BATCH = 32
LR = 1e-4
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(), norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), norm,
])

train_ds_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
eval_ds_full = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
classes = train_ds_full.classes
print("Classes:", classes)

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(train_ds_full), generator=g).tolist()
n = len(perm)
n_train, n_val = int(0.7 * n), int(0.15 * n)
train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

train_loader = DataLoader(Subset(train_ds_full, train_idx), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(Subset(eval_ds_full, val_idx), batch_size=BATCH)
test_loader = DataLoader(Subset(eval_ds_full, test_idx), batch_size=BATCH)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={running_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

test_acc = evaluate(test_loader)
print(f"\nTest accuracy: {test_acc:.4f}")

model.eval()
preds, truths = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        truths += y.tolist()
print(classification_report(truths, preds, target_names=classes))

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "crop_disease_resnet18.pt"))
with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(classes, f, indent=2)
print("Model saved to model/")