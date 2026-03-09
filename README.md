# 🫁 AI-Based Pulmonary Malignancy Detection
**BVRIT Major Project 2025-26 — AI & DS Department**

> Team: Yelagori Sai Kiran · Yendapally Harshitha · Amol Sarkelwad  
> Guide: Dr. S. Pavan Kumar Reddy

---

## 📁 Project Structure

```
lung_cancer_detection/
├── app.py              ← Main Gradio interface + inference logic
├── train.py            ← Model training script
├── requirements.txt    ← Python dependencies
├── model_weights.pth   ← Your trained weights (place here after training)
└── data/               ← Dataset folder (you create this)
    ├── train/
    │   ├── normal/
    │   ├── stage1/
    │   ├── stage2/
    │   ├── stage3/
    │   └── stage4/
    └── val/
        ├── normal/
        ├── stage1/
        ├── stage2/
        ├── stage3/
        └── stage4/
```

---

## ⚙️ Setup (VS Code)

### Step 1 — Create a virtual environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA version of PyTorch from https://pytorch.org

---

## 🚀 Running the App

### Without trained weights (Demo / Random predictions)
```bash
python app.py
```
Open browser at: **http://localhost:7860**

### With trained weights
1. Train the model (see below) to generate `model_weights.pth`
2. Place `model_weights.pth` in the same folder as `app.py`
3. Run `python app.py` — it will auto-load the weights

---

## 🏋️ Training Your Own Model

### Dataset
- Use **LIDC-IDRI** (CT scans) from The Cancer Imaging Archive:  
  https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
- Or **ChestX-ray14** for X-rays:  
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- Organise images into `data/train/` and `data/val/` folders by class.

### Run training
```bash
python train.py --data_dir ./data --epochs 30 --batch_size 16
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Path to dataset |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `16` | Batch size (reduce if OOM) |
| `--lr` | `1e-4` | Initial learning rate |
| `--output` | `model_weights.pth` | Where to save best weights |

---

## 🧠 Model Architecture

- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Head:** Dropout → Linear(2048→256) → ReLU → Dropout → Linear(256→5)
- **Classes:** Normal, Stage I, Stage II, Stage III, Stage IV
- **Training strategy:** Head-only for 5 epochs → full fine-tuning thereafter
- **Input:** 224×224, grayscale converted to 3-channel RGB

---

## 💊 Medication Database

The app includes a curated clinical medication guide for each stage:

| Stage | Primary Treatment | Example Drugs |
|---|---|---|
| Normal | Monitoring only | — |
| Stage I | Surgery ± adjuvant | Osimertinib, Cisplatin+Vinorelbine |
| Stage II | Surgery + chemo | Cisplatin+Pemetrexed, Gemcitabine |
| Stage III | CRT + immunotherapy | Durvalumab, Carboplatin+Paclitaxel |
| Stage IV | Targeted/Immuno/Chemo | Pembrolizumab, Alectinib, Osimertinib |

Medications are selected based on molecular subtype. The app provides general guidelines — an oncologist must finalise the regimen.

---

## ⚠️ Disclaimer

This system is a **research prototype** developed as an academic project. It must **not** be used as a standalone diagnostic tool. All outputs require review and confirmation by a qualified oncologist.

---

## 📚 References

- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- LIDC-IDRI Dataset — The Cancer Imaging Archive.
- NCCN Guidelines for Non-Small Cell Lung Cancer (2024).
- Soria et al. (2018). Osimertinib in Untreated EGFR-Mutated NSCLC. NEJM.
