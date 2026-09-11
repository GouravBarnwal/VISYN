# VISYN
 
### Deep Visual Intelligence for Quality Inspection
 
VISYN is a production-oriented computer vision system for automated industrial visual quality inspection.
 
It learns the visual characteristics of normal samples and detects deviations by comparing incoming images against category-specific reference representations. The system combines deep visual features, patch-level similarity matching, calibrated anomaly scoring, spatial evidence localization, a REST API, a production web interface, CPU-oriented inference, Docker support, and public deployment.
 
## Live System
 
- **Frontend:** https://visyn-chi.vercel.app
- **Backend API:** https://visyn.onrender.com
- **Health endpoint:** https://visyn.onrender.com/health
```text
Vercel Frontend
      │
      ▼
Render FastAPI Service
      │
      ▼
Production Inference Engine
      │
      ├── Category-specific reference bank
      ├── Anomaly scoring
      ├── PASS / REVIEW / FAIL decision
      └── Spatial localization evidence
```
 
---
 
## What VISYN Does
 
Industrial inspection systems frequently need to detect defects or visual deviations when exhaustive labeled examples of every possible defect are unavailable.
 
VISYN approaches this as a **normal-reference comparison problem**. Normal samples are encoded into spatial deep feature representations and stored in reference banks. A new image is transformed using the same production pipeline, compared against the appropriate category reference bank, and converted into a calibrated anomaly score.
 
The final decision is:
 
- **PASS** — score is below the category threshold.
- **REVIEW** — score reaches the threshold but remains inside the review boundary.
- **FAIL** — score reaches the review boundary.
Localization provides spatial evidence about where the visual deviation is concentrated. It is explanatory and does not override the classification decision.
 
---
 
## Full Project Pipeline
 
This is the end-to-end lifecycle of VISYN, from raw data to the deployed public system. It sits one level above the production inference pipeline below — it covers everything done *once*, offline, to arrive at the frozen production configuration.
 
```text
Normal Sample Collection (per category)
            │
            ▼
Train / Development / Evaluation Split
    (80% reference, seed = 42)
            │
            ▼
Feature Extraction
    MobileNetV3-Small → L4 + L8 layers
            │
            ▼
Reference Bank Construction
    (per category, 196 L2-normalized patches/layer)
            │
            ▼
Threshold Calibration
    P99 normal-development threshold
    Review boundary = threshold × 1.10
            │
            ▼
Held-out Evaluation
    AUROC / Average Precision per category
            │
            ▼
Architecture & Threshold Freeze
    (no post-hoc tuning after this point)
            │
            ▼
Backend Implementation (FastAPI)
    /health, /inspect, lazy-loaded reference banks
            │
            ▼
Frontend Implementation (Vite + React + TS)
    upload, camera capture, category selection, overlays
            │
            ▼
Containerization (Docker, CPU-only)
            │
            ▼
Deployment
    Backend → Render
    Frontend → Vercel
            │
            ▼
Live Production System
```
 
Everything from "Architecture & Threshold Freeze" onward is fixed; only the inference pipeline below runs per-request in production.
 
---
 
## Production Pipeline
 
This is what runs on every incoming image at inference time.
 
```text
Input Image
    │
    ▼
Direct Resize to 224 × 224
    │
    ▼
Scale /255 + ImageNet Normalization
    │
    ▼
MobileNetV3-Small
    │
    ├───────────────┐
    ▼               ▼
L4 Features      L8 Features
40 channels      48 channels
14 × 14          14 × 14
    │               │
    ▼               ▼
196 patches      196 patches
    │               │
    ▼               ▼
L2 normalization
    │               │
    └───────┬───────┘
            ▼
Category-specific normal reference bank
            │
            ▼
Patch-level cosine similarity / Euclidean-equivalent distance
            │
            ▼
TOP-5 mean aggregation
            │
            ▼
50/50 raw L4 + L8 fusion
            │
            ▼
Category-specific P99 normal-development threshold
            │
      ┌─────┼─────┐
      ▼     ▼     ▼
    PASS  REVIEW  FAIL
            │
            ▼
Spatial anomaly evidence
            │
            ▼
Heatmap / connected region / bounding box
```
 
### Production scoring configuration
 
| Component | Production configuration |
|---|---|
| Backbone | MobileNetV3-Small |
| Feature layers | L4 + L8 |
| Input | 224 × 224 |
| Patch count per layer | 196 |
| Feature normalization | L2 |
| Distance | Euclidean-equivalent cosine distance |
| Aggregation | TOP-5 mean |
| Feature fusion | Raw 50/50 L4 + L8 |
| Threshold calibration | Category-specific P99 |
| Review boundary | Threshold × 1.10 |
| Localization evidence | P95 patch-distance evidence |
| Reference normal ratio | 80% |
| Random seed | 42 |
 
The production architecture was frozen after evaluation and is intentionally not changed by deployment-time tuning.
 
---
 
## Supported Inspection Categories
 
VISYN currently supports six industrial object categories:
 
- Bottle
- Cable
- Capsule
- Hazelnut
- Metal Nut
- Screw
Each category uses its own normal reference bank and calibrated decision threshold.
 
---
 
## Evaluation
 
The locked evaluation was performed across the six supported categories using a fixed normal reference/development split and held-out defect evaluation data.
 
### Overall performance
 
| Metric | Macro score |
|---|---:|
| AUROC | **0.9511** |
| Average Precision | **0.9525** |
 
### Per-category results
 
| Category | AUROC | AP |
|---|---:|---:|
| Bottle | 1.0000 | 1.0000 |
| Hazelnut | 0.9870 | 0.9819 |
| Cable | 0.9641 | 0.9734 |
| Capsule | 0.9209 | 0.9344 |
| Screw | 0.8463 | 0.8354 |
| Metal Nut | 0.9882 | 0.9900 |
 
These results are reported from the locked evaluation configuration rather than from post-hoc threshold tuning.
 
---
 
## Decision Logic
 
For each category, the normal-development score distribution is used to establish a production threshold at the 99th percentile.
 
Let `T` be the category threshold and `R = 1.10 × T` the review boundary:
 
```text
score < T       → PASS
T ≤ score < R   → REVIEW
score ≥ R       → FAIL
```
 
This creates an explicit intermediate state for samples that should receive additional manual attention instead of forcing every borderline case into a binary decision.
 
---
 
## Localization
 
VISYN also computes spatial anomaly evidence from the L4/L8 patch distances.
 
The localization pipeline uses:
 
1. Patch-level anomaly distances.
2. A P95 evidence threshold.
3. Connected-component analysis.
4. The largest connected anomalous region.
5. A bounding box and region statistics mapped back to the original image.
Localization is deliberately separated from classification. A localization box is explanatory evidence; it does not change PASS, REVIEW, or FAIL.
 
---
 
## Production API
 
The backend is implemented with FastAPI.
 
### Health check
 
```http
GET /health
```
 
Example response:
 
```json
{
  "status": "healthy",
  "engine_loaded": true,
  "categories": [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "screw"
  ]
}
```
 
### Inspection
 
```http
POST /inspect?category=bottle
Content-Type: multipart/form-data
```
 
The request contains the inspection image as a multipart file upload.
 
The response contains:
 
- category
- L4 score
- L8 score
- fused anomaly score
- production threshold
- review threshold
- final decision
- localization method
- bounding box
- center coordinates
- region statistics
---
 
## Frontend
 
The production frontend is a Vite + React + TypeScript application.
 
It provides:
 
- Image upload
- Camera capture
- Category selection
- Production inspection
- PASS / REVIEW / FAIL presentation
- Anomaly score and threshold display
- Localization overlay
- Backend health status
- Upload validation
- Large-image protection
- Responsive desktop/mobile layout
- Reduced-motion accessibility support
- Production error handling
The frontend is deployed on Vercel and communicates with the FastAPI service deployed on Render.
 
---
 
## Repository Structure
 
```text
VISYN/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.tsx
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.ts
│
├── src/
│   ├── api.py
│   ├── production_config.py
│   ├── production_inference.py
│   └── production_localization.py
│
├── artifacts/
│   ├── evaluation/
│   └── production/
│
├── Dockerfile
├── requirements.txt
└── README.md
```
 
---
 
## Local Backend Setup
 
Create and activate a Python virtual environment, then install the backend dependencies:
 
```bash
python -m venv .venv
```
 
Windows PowerShell:
 
```powershell
.venv\Scripts\Activate.ps1
```
 
Install dependencies:
 
```bash
pip install -r requirements.txt
```
 
Start the API locally:
 
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```
 
The local health endpoint is:
 
```text
http://127.0.0.1:8000/health
```
 
---
 
## Local Frontend Setup
 
From the repository root:
 
```bash
cd frontend
npm install
npm run dev
```
 
The Vite development server will provide the local frontend URL shown in the terminal.
 
Production build:
 
```bash
npm run build
```
 
Linting:
 
```bash
npm run lint
```
 
---
 
## Docker
 
VISYN includes a Docker configuration for the backend service and supports CPU-only inference.
 
Build:
 
```bash
docker build -t visyn .
```
 
Run:
 
```bash
docker run -p 8000:8000 visyn
```
 
---
 
## Production Engineering
 
VISYN was designed as an end-to-end engineering project rather than only a model experiment.
 
Important production considerations include:
 
- Category-specific reference banks.
- Lazy loading of reference banks to reduce service memory usage.
- Artifact and configuration validation.
- FastAPI request validation.
- Image decoding and pixel-count limits.
- CORS configuration for local and production frontend origins.
- CPU-compatible inference.
- Docker deployment support.
- Separate classification and localization responsibilities.
- Deterministic evaluation splits using seed 42.
- Frozen production architecture and thresholds.
- Public frontend/backend deployment.
The reference-bank lazy-loading design was specifically used to keep the production service within the available memory budget while retaining all six categories.
 
---
 
## Limitations
 
VISYN is a production-oriented prototype, not a claim of universal defect detection.
 
Known limitations include:
 
- Performance varies by category and defect type.
- Screw and capsule show weaker discrimination than the strongest categories in the locked evaluation.
- Localization is approximate spatial evidence rather than pixel-perfect segmentation.
- The system assumes that the selected category matches the object being inspected.
- Performance depends on the quality and coverage of the normal reference bank.
- New categories require their own normal reference data, reference bank, and threshold calibration.
These limitations are intentionally documented rather than hidden by additional threshold tuning.
 
---
 
## Engineering Decisions
 
Several alternatives were evaluated during development. The production configuration prioritizes stability, interpretability, and deployability over maximizing a single offline metric.
 
The final system uses MobileNetV3-Small rather than a heavier research backbone because the production service is intended to operate within CPU and memory constraints.
 
DINOv2 was retained as a research/benchmark direction but is not part of the frozen production inference path.
 
The final production scorer also deliberately uses raw 50/50 L4/L8 fusion and category-specific calibration rather than adding post-hoc normalization or additional tuning after the architecture freeze.
 
---
 
## Deployment Architecture
 
```text
                         Internet
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Vercel              │
                 │ React + TypeScript  │
                 │ VISYN Frontend      │
                 └──────────┬──────────┘
                            │ HTTPS
                            ▼
                 ┌─────────────────────┐
                 │ Render              │
                 │ FastAPI             │
                 │ VISYN Backend       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Production Engine   │
                 │ MobileNetV3-Small   │
                 │ L4/L8 Reference     │
                 │ Banks + Thresholds  │
                 └─────────────────────┘
```
 
---
 
## Project Status
 
VISYN has completed its core model development, evaluation, productionization, frontend implementation, and public deployment.
 
Current public system:
 
- **Frontend:** deployed and production-ready.
- **Backend:** deployed and production-ready.
- **Inference:** frozen production configuration.
- **Evaluation:** locked final results available in the repository artifacts.
- **Documentation:** maintained alongside the production repository.
---
 
## Author
 
**Gourav Barnwal**
 
VISYN — Deep Visual Intelligence for Quality Inspection
 
Production Computer Vision • Anomaly Detection • Visual Inspection • Machine Learning Engineering
 
