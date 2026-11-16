import json, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from common import *

EPOCHS, LR, BS = 3, 1e-5, 32
classes = json.load(open("model/classes.json"))

MAP = {
    "Apple_leaf": "Apple___healthy",
    "Apple_rust_leaf": "Apple___Cedar_apple_rust",
    "Apple_Scab_Leaf": "Apple___Apple_scab",
    "Bell_pepper_leaf": "Pepper,_bell___healthy",
    "Bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry_leaf": "Blueberry___healthy",
    "Cherry_leaf": "Cherry_(including_sour)___healthy",
    "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
    "grape_leaf": "Grape___healthy",
    "grape_leaf_black_rot": "Grape___Black_rot",
    "Peach_leaf": "Peach___healthy",
    "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight",
    "Raspberry_leaf": "Raspberry___healthy",
    "Soyabean_leaf": "Soybean___healthy",
}

# Augmentation that mimics field photos: crops, rotation, colour and lighting changes
field_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(), norm])

class FieldSet(Dataset):
    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), y

# Only PlantDoc TRAIN images. The test folder is never used here.
raw = datasets.ImageFolder("data/plantdoc/train")
samples = [(p, classes.index(MAP[raw.classes[y]]))
           for p, y in raw.samples if raw.classes[y] in MAP]
print("Field training images:", len(samples))

perm = list(range(len(samples)))
random.Random(SEED).shuffle(perm)
n_val = int(0.1 * len(samples))
val_s = [samples[i] for i in perm[:n_val]]
train_s = [samples[i] for i in perm[n_val:]]
print("Train:", len(train_s), "Validation:", len(val_s))

train_loader = DataLoader(FieldSet(train_s, field_tf), batch_size=BS, shuffle=True)
val_loader = DataLoader(FieldSet(val_s, eval_tf), batch_size=64)

model = make_model(len(classes))
model.load_state_dict(torch.load("model/fedavg_iid.pt", map_location=device))
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    preds, truths = predict_all(model, val_loader)
    acc = sum(p == t for p, t in zip(preds, truths)) / len(truths)
    print(f"Epoch {epoch+1}/{EPOCHS}  val_acc={acc:.4f}")

torch.save(model.state_dict(), "model/field_finetuned.pt")
print("Saved model/field_finetuned.pt")