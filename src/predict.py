import json, sys, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

classes = json.load(open("model/classes.json"))
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model/fedavg_iid.pt", map_location="cpu"))
model.eval()

def predict(path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = model(x).softmax(1)[0]
    i = p.argmax().item()
    return classes[i], p[i].item()

if __name__ == "__main__":
    name, conf = predict(sys.argv[1])
    print(f"Predicted: {name}  ({conf:.1%})")