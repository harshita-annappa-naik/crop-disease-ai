Explainable Federated Crop Disease Detection

Classifies 38 plant disease classes from leaf images using ResNet18, trained centrally and with federated averaging (FedAvg) across three simulated farms, with Grad-CAM explanations.

Setup
bash
git clone https://github.com/harshita-annappa-naik/crop-disease-ai.git
cd crop-disease-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Run every command from the project root folder, not from inside src/.

Use the trained model

The recommended model is model/fedavg_iid.pt, with class names in model/classes.json.

bash
python src/predict.py path/to/leaf.jpg
Reproduce the results

The training data is not in this repo. Download PlantVillage (color images) and place the class folders in data/plantvillage/. Then run:

bash
python src/train_central.py
python src/fedavg.py iid
python src/fedavg.py noniid
python src/gradcam.py model/fedavg_iid.pt
python src/make_results.py

Field test (optional): download PlantDoc into data/plantdoc/ and run:

bash
python src/eval_field.py model/fedavg_iid.pt
Results
Model	Test accuracy (PlantVillage)
Centralised ResNet18	97.55%
FedAvg IID	99.11%
FedAvg non-IID	98.89%

Field test (PlantDoc, 159 images): FedAvg 22.64%, centralised 25.79%, FedAvg with field fine-tuning 45.28%.

Limitations
300 images per class, 224×224 input, CPU training.
Random image-level split, which can overstate accuracy because PlantVillage contains near-duplicate images.
The environmental (temperature, humidity, soil moisture) branch is future work; no paired dataset was available.
FedProx and MC-dropout uncertainty are not implemented.