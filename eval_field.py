import json, sys, torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import accuracy_score, classification_report
from common import *

weights = sys.argv[1]
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

test = datasets.ImageFolder("data/plantdoc/test", transform=eval_tf)
keep = [i for i, (_, y) in enumerate(test.samples) if test.classes[y] in MAP]
print("Field images scored:", len(keep), "of", len(test))
truth = [MAP[test.classes[test.samples[i][1]]] for i in keep]

model = make_model(len(classes))
model.load_state_dict(torch.load(weights, map_location=device))
preds, _ = predict_all(model, DataLoader(Subset(test, keep), batch_size=64))
pred_names = [classes[p] for p in preds]

acc = accuracy_score(truth, pred_names)
print(classification_report(truth, pred_names, zero_division=0))
print("Field accuracy:", round(acc, 4))
json.dump({"weights": weights, "field_accuracy": acc, "n": len(keep)},
          open(f"results/field_{weights.split('/')[-1].replace('.pt', '')}.json", "w"), indent=2)