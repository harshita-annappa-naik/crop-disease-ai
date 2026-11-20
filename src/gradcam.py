import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets
from common import *

weights = sys.argv[1]  # e.g. model/fedavg_iid.pt
base, classes, train_idx, test_idx = load_splits()
test_ds = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

model = make_model(len(classes))
model.load_state_dict(torch.load(weights, map_location=device))
model.eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

correct, wrong = [], []
for i in test_idx:
    x, y = test_ds[i]
    with torch.no_grad():
        probs = model(x.unsqueeze(0).to(device)).softmax(1)[0]
    pred = probs.argmax().item()
    item = (x, y, pred, probs[pred].item())
    if pred == y and len(correct) < 4:
        correct.append(item)
    elif pred != y and len(wrong) < 4:
        wrong.append(item)
    if len(correct) == 4 and len(wrong) == 4:
        break

samples = correct + wrong
fig, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 7))
for j, (x, y, pred, conf) in enumerate(samples):
    grey = cam(input_tensor=x.unsqueeze(0).to(device), targets=None)[0]
    rgb = (x.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
    overlay = show_cam_on_image(rgb, grey, use_rgb=True)
    axes[0, j].imshow(rgb); axes[0, j].axis("off")
    axes[1, j].imshow(overlay); axes[1, j].axis("off")
    status = "OK" if pred == y else "WRONG"
    axes[1, j].set_title(f"{status}\ntrue: {classes[y]}\npred: {classes[pred]} ({conf:.0%})", fontsize=6)
plt.tight_layout()
plt.savefig("results/gradcam_samples.png", dpi=110)
print("Saved results/gradcam_samples.png")