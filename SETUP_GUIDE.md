# 🚀 SmartSched - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [File Checklist](#file-checklist)
4. [Training Models](#training-models)
5. [Running Demos](#running-demos)
6. [Docker Deployment](#docker-deployment)
7. [Troubleshooting](#troubleshooting)

---

## ✅ Prerequisites

### Required Software
- Python 3.11 or higher
- pip (Python package manager)
- Git
- (Optional) Docker Desktop

### Required Python Packages
All listed in `requirements.txt`:
```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
tensorflow==2.13.0
flask==2.3.3
flask-cors==4.0.0
```

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Train Models
```bash
# Quick training (3-5 minutes)
python train_models.py --quick

# Or full training with Borg data (if available)
python train_models.py --use-borg
```

### Step 3: Run Demo
```bash
# Complete demo with all features
python demo_advanced.py

# Or simple demo
python demo_simple.py
```

---

## 📂 File Checklist

Ensure you have all these files in `D:\SmartSched\`:

### ✅ Core Files (ROOT directory)
```
D:\SmartSched\
├── requirements.txt              ✅
├── README.md                     ✅
├── SETUP_GUIDE.md               ✅
├── Dockerfile                    ✅ NEW
├── docker-compose.yml           ✅ NEW
├── .gitignore                   ✅
│
├── smart_scheduler.py           ✅
├── visualization.py             ✅
├── data_loader.py               ✅
├── train_models.py              ✅
│
├── main_demo.py                 ✅
├── demo_simple.py               ✅
├── demo_borg.py                 ✅
├── demo_advanced.py             ✅ NEW
│
├── app.py                       ✅
├── rl_scheduler.py              ✅ NEW
└── multicore_scheduler.py       ✅ NEW
```

### ✅ Models Directory
```
D:\SmartSched\models\
├── burst_predictor.py           ✅
├── lstm_predictor.py            ✅
├── rl_agent.py                  ✅
│
├── random_forest_model.pkl      ✅ (after training)
├── gradient_boosting_model.pkl  ✅ (after training)
├── lstm_model.h5                ✅ (after training)
├── rl_agent.pkl                 ✅ (after training)
└── scaler.pkl                   ✅ (after training)
```

### ✅ Data Directory (Optional)
```
D:\SmartSched\data\
└── borg_traces_data.csv         ✅ (if you have it)
```

---

## 🎓 Training Models

### Option 1: Quick Training (Synthetic Data)
**Use this for testing - takes 3-5 minutes**
```bash
python train_models.py --quick
```

### Option 2: Full Training (Synthetic Data)
**Better accuracy - takes 5-10 minutes**
```bash
python train_models.py
```

### Option 3: Real Borg Data Training
**BEST option if you have Borg data - takes 10-15 minutes**
```bash
python train_models.py --use-borg
```

### Option 4: Specific Models Only
```bash
# Train only Random Forest
python train_models.py --models rf

# Train RF and LSTM
python train_models.py --models rf lstm

# Train all except LSTM (if TensorFlow issues)
python train_models.py --models rf gb rl
```

### Verify Training Success
After training, check if these files exist:
```bash
dir models\random_forest_model.pkl
dir models\gradient_boosting_model.pkl
dir models\lstm_model.h5
dir models\rl_agent.pkl
```

---

## 🎬 Running Demos

### Demo 1: Simple Demo (No Visualizations)
**Best for quick testing**
```bash
python demo_simple.py
```
- Text-based output
- Shows ML predictions
- Algorithm comparison
- Takes 10-20 seconds

### Demo 2: Main Demo (With Graphs)
**Best for expo presentation**
```bash
python main_demo.py
```
- Interactive menu
- Choose workload type
- 3 beautiful graphs:
  - Gantt chart
  - Algorithm comparison
  - ML prediction accuracy
- Takes 30-60 seconds

### Demo 3: Advanced Features Demo ⭐ NEW
**Shows ALL new features**
```bash
python demo_advanced.py
```
- RL Meta-Scheduler
- Multi-Core scheduling
- ML predictions
- Comprehensive comparison
- Takes 1-2 minutes

### Demo 4: Real Borg Data Demo
**If you have Google Borg data**
```bash
python demo_borg.py
```
- Loads real datacenter traces
- Validates on production data
- Shows ML on real workloads

### Demo 5: RL Meta-Scheduler Only
**Focus on AI algorithm selection**
```bash
python rl_scheduler.py
```
- Shows RL agent decision-making
- Explains reasoning
- Compares with other algorithms

### Demo 6: Multi-Core Only
**Focus on parallel execution**
```bash
python multicore_scheduler.py
```
- 4-core simulation
- Memory management
- Per-core utilization

### Demo 7: Web Interface
**For interactive demo**
```bash
python app.py
# Open: http://localhost:5000/simple
```

---

## 🐳 Docker Deployment

### Prerequisites
- Docker Desktop installed
- Docker running

### Quick Docker Start
```bash
# Build and run
docker-compose up

# Access web interface
# Open: http://localhost:5000/simple
```

### Docker Commands

#### Build Image
```bash
docker build -t smartsched .
```

#### Run Container
```bash
# Web app
docker run -p 5000:5000 smartsched

# Run demo
docker run smartsched python main_demo.py

# Train models
docker run smartsched python train_models.py --quick

# Advanced demo
docker run smartsched python demo_advanced.py

# Shell access
docker exec -it smartsched-app bash
```

#### Docker Compose Commands
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose up --build
```

---

## 🔧 Troubleshooting

### Issue 1: Module Not Found Errors

**Error**: `ModuleNotFoundError: No module named 'smart_scheduler'`

**Solution**:
```bash
# Check your current directory
cd D:\SmartSched

# Run from project root
python demo_simple.py
```

### Issue 2: ML Models Not Loading

**Error**: `⚠️  ML predictor not available`

**Solution**:
```bash
# Train models first
python train_models.py --quick

# Verify models exist
dir models\*.pkl
dir models\*.h5
```

### Issue 3: TensorFlow/LSTM Issues

**Error**: TensorFlow installation or LSTM training fails

**Solution**:
```bash
# Train without LSTM
python train_models.py --models rf gb rl

# Or install TensorFlow separately
pip install tensorflow==2.13.0

# For GPU support (optional)
pip install tensorflow-gpu==2.13.0
```

### Issue 4: Flask/Web App Issues

**Error**: `ModuleNotFoundError: No module named 'flask_cors'`

**Solution**:
```bash
pip install flask-cors
```

**Error**: Port 5000 already in use

**Solution**:
```bash
# Use different port
# Edit app.py, change last line to:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue 5: Matplotlib Display Issues

**Error**: Plots don't show

**Solution**:
```bash
# Windows
pip install pyqt5

# Or use non-interactive backend
# Add to top of demo file:
import matplotlib
matplotlib.use('Agg')
```

### Issue 6: Borg Data Issues

**Error**: `❌ File not found: data/borg_traces_data.csv`

**Solution**:
```bash
# Place your Borg CSV in data folder
mkdir data
# Copy borg_traces_data.csv to data\

# Or run without Borg data
python train_models.py  # Uses synthetic data
```

### Issue 7: Memory Issues

**Error**: Out of memory during training

**Solution**:
```bash
# Use quick mode with fewer samples
python train_models.py --quick

# Or reduce samples
python train_models.py --samples 5000 --lstm-epochs 20
```

### Issue 8: Docker Issues

**Error**: Docker build fails

**Solution**:
```bash
# Check Docker is running
docker --version

# Try simpler Dockerfile
# Remove TensorFlow if issues:
# Edit requirements.txt, comment out:
# tensorflow==2.13.0

# Rebuild
docker-compose up --build
```

---

## 📊 Verify Installation

### Complete Verification Script

Create `verify_installation.py`:

```python
"""Verify SmartSched installation"""
import os
import sys

print("="*70)
print("🔍 SmartSched Installation Verification")
print("="*70)

# Check Python version
print(f"\n✅ Python: {sys.version}")

# Check critical files
files = [
    'smart_scheduler.py',
    'visualization.py', 
    'train_models.py',
    'rl_scheduler.py',
    'multicore_scheduler.py',
    'models/burst_predictor.py',
    'models/lstm_predictor.py',
    'models/rl_agent.py'
]

print("\n📁 Critical Files:")
for file in files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")

# Check packages
packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn',
    'sklearn', 'tensorflow', 'flask'
]

print("\n📦 Python Packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except:
        print(f"   ❌ {pkg}")

# Check trained models
print("\n🤖 Trained Models:")
models = [
    'models/random_forest_model.pkl',
    'models/gradient_boosting_model.pkl',
    'models/lstm_model.h5',
    'models/rl_agent.pkl'
]

any_trained = False
for model in models:
    exists = os.path.exists(model)
    status = "✅" if exists else "⚠️"
    print(f"   {status} {model}")
    if exists:
        any_trained = True

if not any_trained:
    print("\n   ⚠️  No trained models found.")
    print("   Run: python train_models.py --quick")

print("\n" + "="*70)
print("✨ Verification Complete!")
print("="*70)
```

Run it:
```bash
python verify_installation.py
```

---

## 🎯 Recommended Setup for Expo

### Day Before Expo:
1. ✅ Verify all files present
2. ✅ Train models: `python train_models.py --use-borg`
3. ✅ Test all demos: `python demo_advanced.py`
4. ✅ Test web interface: `python app.py`
5. ✅ Take screenshots of outputs
6. ✅ Print README.md and cheat sheet

### Expo Day Setup:
1. ✅ Laptop fully charged
2. ✅ Virtual environment activated
3. ✅ Models trained and verified
4. ✅ Have `demo_advanced.py` ready to run
5. ✅ Web app tested and working
6. ✅ Backup: USB with full code

### For Judges Demo:
```bash
# Activate environment
venv\Scripts\activate

# Run complete demo
python demo_advanced.py

# Or if time limited
python demo_simple.py
```

---

## 🆘 Emergency Quick Fix

If everything breaks before expo:

```bash
# 1. Fresh start
cd D:\SmartSched
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate

# 2. Reinstall
pip install --upgrade pip
pip install -r requirements.txt

# 3. Quick train
python train_models.py --quick

# 4. Test simple demo
python demo_simple.py
```

If still broken, use `demo_simple.py` - it's self-contained!

---

## 📞 Support

If you need help:
1. Check this guide
2. Review error messages carefully
3. Check GitHub issues
4. Email: saminenidevipragna@gmail.com

---

**✨ You're ready to go! Good luck at the expo! 🏆**