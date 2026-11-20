import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from torchvision import datasets
from common import *

EPOCHS, LR, BS = 4, 1e-4, 32
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

base, classes, train_idx, test_idx = load_splits()
json.dump(classes, open("model/classes.json", "w"), indent=2)
train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
test_ds = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BS, shuffle=True)
test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=64)

print("Classes:", len(classes))
print("Train:", len(train_idx), "Test:", len(test_idx))

model = make_model(len(classes))
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    preds, truths = predict_all(model, test_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  test_acc={accuracy_score(truths, preds):.4f}")

preds, truths = predict_all(model, test_loader)
acc = accuracy_score(truths, preds)
report = classification_report(truths, preds, target_names=classes, output_dict=True)
print(classification_report(truths, preds, target_names=classes, zero_division=0))
print("Test accuracy:", round(acc, 4))

torch.save(model.state_dict(), "model/central.pt")
json.dump({"method": "centralised", "test_accuracy": acc, "report": report},
          open("results/central.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(14, 14))
ConfusionMatrixDisplay.from_predictions(truths, preds, display_labels=classes,
                                        xticks_rotation=90, ax=ax, colorbar=False)
plt.tight_layout()
plt.savefig("results/confusion_central.png", dpi=100)
print("Saved model/central.pt, results/central.json, results/confusion_central.png")