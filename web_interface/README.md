# Scanvidence Standalone Web Interface & Clinical PACS Server

A completely standalone, zero-external-dependency distribution of the Scanvidence web platform, containing both frontend and backend components.

---

## 📁 Directory Structure

```
web_interface/
├── index.html            # Executive Clinical Landing Page
├── workstation.html      # Clinical PACS 4-Quadrant Slicer & Triage Workstation
├── benchmarks.html       # Peer-Reviewed Scientific Benchmarks & Audit Defense
├── server.py             # Zero-dependency Python HTTP server & REST APIs
├── run.sh                # 1-Click launcher script
└── assets/
    ├── slices/           # 13 Real BraTS Patient 3D Multi-Planar Slice Stacks & Metadata
    ├── explainability/   # XAI Saliency and Gradient Attribution Maps
    └── nifti/            # Real 3D NIfTI (.nii.gz) Tumor Volumes
```

---

## 🚀 Quick Start (Zero Dependencies)

Run directly with standard Python 3 (no pip installations required):

```bash
# Option 1: Using the launcher script
./run.sh 8000

# Option 2: Running directly with Python
python3 server.py 8000
```

---

## 🌐 Local Access Endpoints

* **Landing Page**: [http://localhost:8000](http://localhost:8000)
* **PACS Workstation (13 3D Cases + Live Execution)**: [http://localhost:8000/workstation](http://localhost:8000/workstation)
* **Benchmarks & Audit**: [http://localhost:8000/benchmarks](http://localhost:8000/benchmarks)
* **Health API**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
* **Patients API**: [http://localhost:8000/api/patients](http://localhost:8000/api/patients)
