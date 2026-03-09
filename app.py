"""
Lung Cancer Detection System
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os

# ─────────────────────────────────────────────
# 1. MODEL DEFINITION
# ─────────────────────────────────────────────

class LungCancerCNN(nn.Module):
    """
    CNN model based on ResNet-50 backbone for lung cancer staging.
    In production, load your trained weights via: model.load_state_dict(torch.load('weights.pth'))
    """
    def __init__(self, num_classes=5):
        super(LungCancerCNN, self).__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Load model (or use random weights for demo)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LungCancerCNN(num_classes=5).to(DEVICE)

# ── Load weights if available ──────────────────
WEIGHTS_PATH = "model_weights.pth"
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print(f"[INFO] Loaded weights from {WEIGHTS_PATH}")
else:
    print("[INFO] No weights file found. Running in DEMO MODE with random predictions.")

model.eval()

# ─────────────────────────────────────────────
# 2. IMAGE PREPROCESSING
# ─────────────────────────────────────────────

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # handles grayscale X-ray/CT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASSES = [
    "No Cancer Detected (Normal)",
    "Stage I  — Early Stage",
    "Stage II — Locally Early",
    "Stage III — Locally Advanced",
    "Stage IV — Metastatic"
]

# ─────────────────────────────────────────────
# 3. MEDICATION DATABASE
# ─────────────────────────────────────────────

MEDICATION_GUIDE = {
    "No Cancer Detected (Normal)": {
        "summary": "No malignancy detected in the provided scan.",
        "recommendations": [
            "✅ Continue routine annual chest screening (especially if smoker ≥30 pack-years)",
            "✅ Maintain a healthy lifestyle: quit smoking, balanced diet, regular exercise",
            "✅ Follow up with pulmonologist if symptoms (persistent cough, breathlessness) occur",
        ],
        "medications": "No cancer-directed therapy required at this time.",
        "follow_up": "Annual low-dose CT screening recommended for high-risk individuals.",
        "color": "#22c55e"
    },
    "Stage I  — Early Stage": {
        "summary": "Tumor confined to the lung. Excellent prognosis with curative intent possible.",
        "recommendations": [
            "🏥 Primary treatment: Surgical resection (lobectomy / VATS lobectomy)",
            "💊 Adjuvant chemotherapy if high-risk features present",
            "🔬 Molecular profiling: EGFR, ALK, ROS1, PD-L1 testing",
            "☢️ SBRT (Stereotactic Body Radiotherapy) if surgery not feasible",
        ],
        "medications": (
            "• Adjuvant Osimertinib (EGFR+): 80 mg/day orally\n"
            "• Adjuvant Atezolizumab (PD-L1+): 1200 mg IV q3w × 16 cycles\n"
            "• Cisplatin + Vinorelbine (standard adjuvant chemo if needed)"
        ),
        "follow_up": "CT chest every 6 months × 2 years, then annually.",
        "color": "#84cc16"
    },
    "Stage II — Locally Early": {
        "summary": "Tumor with ipsilateral lymph node involvement. Curative resection still possible.",
        "recommendations": [
            "🏥 Surgery: Lobectomy with mediastinal lymph node dissection",
            "💊 Adjuvant chemotherapy: Platinum-doublet × 4 cycles",
            "☢️ PORT (Post-Operative Radiotherapy) if incomplete resection",
            "🔬 Biomarker testing mandatory (EGFR, ALK, ROS1, KRAS, PD-L1)",
        ],
        "medications": (
            "• Cisplatin 75 mg/m² (Day 1) + Pemetrexed 500 mg/m² (Day 1) q21d × 4 cycles [Non-squamous]\n"
            "• Cisplatin 75 mg/m² + Gemcitabine 1250 mg/m² (Days 1,8) q21d × 4 cycles [Squamous]\n"
            "• Osimertinib 80 mg/day (if EGFR exon 19/21 mutation)"
        ),
        "follow_up": "CT chest + abdomen every 3–6 months × 2 years, then annually.",
        "color": "#f59e0b"
    },
    "Stage III — Locally Advanced": {
        "summary": "Bulky mediastinal disease. Multimodal approach required.",
        "recommendations": [
            "💊+☢️ Concurrent chemoradiation (CRT) — standard of care for unresectable IIIA/IIIB",
            "🏥 Surgery considered only for select resectable Stage IIIA cases",
            "🧬 Durvalumab consolidation immunotherapy after CRT (PD-L1 ≥1%)",
            "🔬 Full biomarker panel essential before treatment planning",
        ],
        "medications": (
            "• Carboplatin AUC 5 (Day 1) + Paclitaxel 45 mg/m² weekly (concurrent with RT 60 Gy)\n"
            "• Durvalumab (Imfinzi) 10 mg/kg IV q2w × 12 months (consolidation)\n"
            "• Osimertinib (if EGFR+) or Alectinib (if ALK+) as alternative targeted approach"
        ),
        "follow_up": "CT chest q3 months × 1 year, then q6 months.",
        "color": "#f97316"
    },
    "Stage IV — Metastatic": {
        "summary": "Distant metastasis present. Treatment is palliative/systemic. Personalized therapy critical.",
        "recommendations": [
            "🧬 Biomarker-driven first-line therapy is the gold standard",
            "💊 Targeted therapy if actionable mutation (EGFR, ALK, ROS1, BRAF, MET, RET, KRAS G12C)",
            "🛡️ Immunotherapy (anti-PD-1/PD-L1) for high PD-L1 expressors without driver mutations",
            "💊 Platinum-based chemotherapy ± immunotherapy for others",
            "🩺 Palliative care and symptom management essential",
            "🏥 Clinical trials should be considered",
        ],
        "medications": (
            "EGFR+ (Exon 19 del / L858R): Osimertinib 80 mg/day\n"
            "ALK+: Alectinib 600 mg BID or Lorlatinib 100 mg/day\n"
            "ROS1+: Crizotinib 250 mg BID or Entrectinib 600 mg/day\n"
            "KRAS G12C+: Sotorasib 960 mg/day or Adagrasib 600 mg BID\n"
            "PD-L1 ≥50%, no driver mutation: Pembrolizumab 200 mg IV q3w\n"
            "PD-L1 <50%, no driver mutation: Carboplatin + Pemetrexed + Pembrolizumab\n"
            "Bone mets: Zoledronic acid 4 mg IV q4w + Denosumab 120 mg SC q4w\n"
            "Brain mets (EGFR+): Osimertinib preferred (CNS penetrant)"
        ),
        "follow_up": "CT chest/abdomen/pelvis q6-8 weeks during active therapy. Brain MRI q3 months.",
        "color": "#ef4444"
    }
}

# ─────────────────────────────────────────────
# 4. INFERENCE FUNCTION
# ─────────────────────────────────────────────

def predict(image: Image.Image) -> tuple:
    """Run inference and return (class_index, probabilities_dict)."""
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, {cls: float(p) for cls, p in zip(CLASSES, probs)}


def analyze_scan(xray_img, ct_img, patient_name, patient_age, scan_type):
    """
    Main analysis function called by Gradio interface.
    Uses X-ray if provided, else CT. If both provided, averages predictions.
    """
    if xray_img is None and ct_img is None:
        return (
            "⚠️ Please upload at least one scan (X-Ray or CT Scan).",
            None, "", "", "", ""
        )

    time.sleep(0.5)   # simulate processing

    predictions = []
    prob_dicts   = []

    if xray_img is not None:
        idx, probs = predict(xray_img)
        predictions.append(idx)
        prob_dicts.append(probs)

    if ct_img is not None:
        idx, probs = predict(ct_img)
        predictions.append(idx)
        prob_dicts.append(probs)

    # Average probabilities across available scans
    avg_probs = {}
    for cls in CLASSES:
        avg_probs[cls] = float(np.mean([pd[cls] for pd in prob_dicts]))

    final_class = CLASSES[int(np.argmax([avg_probs[c] for c in CLASSES]))]
    confidence  = avg_probs[final_class] * 100

    guide = MEDICATION_GUIDE[final_class]

    # ── Build outputs ──────────────────────────
    # 1. Diagnosis result
    diagnosis_html = f"""
<div style="background:{guide['color']}22; border-left:5px solid {guide['color']};
            padding:20px; border-radius:8px; margin:10px 0;">
  <h2 style="color:{guide['color']}; margin:0 0 8px 0;">
    🔬 {final_class}
  </h2>
  <p style="margin:4px 0;"><b>Patient:</b> {patient_name or 'N/A'} &nbsp;|&nbsp;
     <b>Age:</b> {patient_age or 'N/A'} &nbsp;|&nbsp;
     <b>Scan type:</b> {scan_type}</p>
  <p style="margin:4px 0;"><b>Model Confidence:</b> {confidence:.1f}%</p>
  <p style="margin:8px 0 0 0; color:#555;">{guide['summary']}</p>
</div>
"""

    # 2. Probability bar chart data (for Gradio Label component)
    label_output = {cls: float(avg_probs[cls]) for cls in CLASSES}

    # 3. Clinical recommendations
    recs = "\n".join(guide["recommendations"])

    # 4. Medications
    meds = guide["medications"]

    # 5. Follow-up
    follow = guide["follow_up"]

    # 6. Disclaimer
    disclaimer = (
        "⚕️ DISCLAIMER: This system is a research prototype (BVRIT Major Project 2025-26). "
        "All outputs must be reviewed and confirmed by a qualified oncologist before any "
        "clinical decision. Do NOT use as a sole diagnostic tool."
    )

    return diagnosis_html, label_output, recs, meds, follow, disclaimer


# ─────────────────────────────────────────────
# 5. GRADIO INTERFACE
# ─────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate"
)

with gr.Blocks(theme=THEME, title="Lung Cancer Detection") as demo:
    # Header
    gr.HTML("""
    <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#1e3a5f,#2563eb);
                border-radius:12px; color:white; margin-bottom:20px;">
      <h1 style="margin:0; font-size:2rem;">🫁 AI-Based Pulmonary Malignancy Detection</h1>
      <p style="margin:8px 0 0 0; font-size:1rem; opacity:0.85;">
        Early Detection of Lung Cancer using CNN — BVRIT Major Project 2025-26
      </p>
      <p style="margin:4px 0 0 0; font-size:0.85rem; opacity:0.7;">
        Guide: Dr. S. Pavan Kumar Reddy &nbsp;|&nbsp;
        Team: Sai Kiran · Harshitha · Amol
      </p>
    </div>
    """)

    with gr.Row():
        # ── LEFT PANEL: Inputs ──────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Patient Details")
            patient_name = gr.Textbox(label="Patient Name", placeholder="e.g. John Doe")
            patient_age  = gr.Number(label="Age", minimum=1, maximum=120, value=55)
            scan_type    = gr.Radio(
                ["X-Ray Only", "CT Scan Only", "Both X-Ray & CT Scan"],
                label="Scan Type Submitted", value="Both X-Ray & CT Scan"
            )

            gr.Markdown("### 🖼️ Upload Scans")
            xray_input = gr.Image(label="Chest X-Ray", type="pil", height=200,
                                  image_mode="L")
            ct_input   = gr.Image(label="Chest CT Scan", type="pil", height=200,
                                  image_mode="L")

            analyze_btn = gr.Button("🔍 Analyze Scans", variant="primary", size="lg")

        # ── RIGHT PANEL: Results ────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            diagnosis_out = gr.HTML(label="Diagnosis")
            prob_out      = gr.Label(label="Stage Probabilities", num_top_classes=5)

            gr.Markdown("### 🩺 Clinical Recommendations")
            recs_out    = gr.Textbox(label="Recommended Actions", lines=6,
                                     interactive=False)

            gr.Markdown("### 💊 Suggested Medications / Therapy")
            meds_out    = gr.Textbox(label="Medication Protocol", lines=8,
                                     interactive=False)

            gr.Markdown("### 📅 Follow-Up Schedule")
            follow_out  = gr.Textbox(label="Follow-Up Plan", lines=2,
                                     interactive=False)

            disclaimer_out = gr.Textbox(label="⚠️ Medical Disclaimer", lines=3,
                                        interactive=False)

    # Wire up button
    analyze_btn.click(
        fn=analyze_scan,
        inputs=[xray_input, ct_input, patient_name, patient_age, scan_type],
        outputs=[diagnosis_out, prob_out, recs_out, meds_out, follow_out, disclaimer_out]
    )

    # Example images section
    gr.Markdown("---")
    gr.Markdown(
        "**How to use:** Upload a chest X-ray and/or CT scan image, fill in patient details, "
        "then click **Analyze Scans**. The CNN model will classify the scan into one of 5 "
        "categories (Normal + 4 cancer stages) and provide corresponding clinical guidance."
    )

# ─────────────────────────────────────────────
# 6. LAUNCH
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set share=True to get a public link
        show_error=True
    )