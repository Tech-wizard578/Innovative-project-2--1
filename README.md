# 🚀 SmartSched: An AI-Powered Process Scheduler

SmartSched is an advanced process scheduling simulator that leverages machine learning and reinforcement learning to optimize scheduling decisions. It is designed to compare traditional scheduling algorithms against "smart" hybrid algorithms that use ML models to predict process burst times.

Furthermore, it includes a Q-learning agent that analyzes the incoming workload and dynamically selects the best scheduling policy to minimize metrics like Average Waiting Time.

This project includes:
* An interactive **Flask Web App** (`app.py`) for visual comparison.
* A terminal-based demo (`main_demo.py`).
* A full training pipeline (`train_models.py`) that learns from real-world **Google Borg** cluster data.

---

## Core Features

* **ML Burst Time Prediction:** Uses **Random Forest**, **Gradient Boosting**, and **LSTM (TensorFlow/Keras)** models to predict a process's burst time based on its features.
* **Data-Driven Models:** The ML predictors are trained on a processed dataset from **Google's Borg cluster**, allowing them to learn realistic process behavior.
* **Reinforcement Learning Agent:** A Q-learning agent (`models/rl_agent.py`) is trained to analyze workload characteristics and select the optimal scheduling policy.
* **Advanced Schedulers:** Implements a full suite of schedulers for comparison:
    * First-Come-First-Served (FCFS)
    * Shortest Job First (SJF)
    * Shortest Remaining Time First (SRTF)
    * Priority Scheduling
    * Round Robin (RR)
    * **SMART\_HYBRID:** An ML-powered hybrid scheduler that uses model predictions to optimize decisions.
* **Interactive Web App:** A **Flask** app (`app.py`) provides a simple web interface to run simulations, compare algorithms, and visualize results.
* **Comprehensive Visualization:** Generates Plotly Gantt charts, performance metric comparisons (Avg. Wait/Turnaround/Response Time), and ML prediction accuracy plots.

---

## ⚙️ Project Architecture

The project follows a clear Data -> Train -> Simulate pipeline.

1.  **Data Processing (`data_loader.py`)**
    * The data loader script takes raw Google Borg data (e.g., `data/borg_traces_data.csv`, not included in the repo) and processes it.
    * It cleans, samples, and feature-engineers the data for training.

2.  **Model Training (`train_models.py`)**
    * This script loads the processed data.
    * It trains multiple ML predictors:
        * **Random Forest** (`models/random_forest_model.pkl`)
        * **Gradient Boosting** (`models/gradient_boosting_model.pkl`)
        * **LSTM** (`models/lstm_model.h5` and `models/scaler.pkl`)
    * It then trains the **Q-learning Agent** (`models/rl_agent.pkl`) by running thousands of simulations.

3.  **Simulation & Application (`app.py` / `main_demo.py`)**
    * The `SmartScheduler` class (`smart_scheduler.py`) is the core engine. On initialization, it pre-loads the trained ML models.
    * The **Flask App** (`app.py`) or Terminal Demo (`main_demo.py`) creates a synthetic workload.
    * It then runs this workload through each of the scheduler's algorithms.
    * Finally, it uses `visualization.py` to plot the results and metrics from each run.

---

## 🔧 Technology Stack

* **Simulation:** Python 3.x
* **Web App:** `flask`, `flask-cors`
* **Data & ML:** `pandas`, `numpy`, `scikit-learn` (for Random Forest, Gradient Boosting & Scaler)
* **Deep Learning:** `tensorflow`, `keras` (for LSTM)
* **Reinforcement Learning:** `gym`, `stable-baselines3` (as dependencies, though core agent is custom Q-Learning/DQN)
* **Visualization:** `plotly`, `matplotlib`, `seaborn`

*(See `requirements.txt` for a full list of dependencies)*

---

## 🚀 How to Run

### Step 1: Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Tech-wizard578/innovative-project-2--1.git](https://github.com/Tech-wizard578/innovative-project-2--1.git)
    cd innovative-project-2--1
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data & Model Training (One-Time Setup)

You must train the models before you can run the simulations.

1.  **Process the Data (Optional)**
    *(This project looks for `data/borg_traces_data.csv`. If you do not have this file, the training script will fall back to using synthetic data).*

2.  **Train All Models**
    This is the most important step. This script will train the RF, GB, LSTM, and RL Agent models.
    ```bash
    python train_models.py
    ```
    This will create the following essential files in the `models/` directory:
    * `random_forest_model.pkl`
    * `gradient_boosting_model.pkl`
    * `lstm_model.h5`
    * `scaler.pkl`
    * `rl_agent.pkl`

### Step 3: Run the Application

You have two ways to run the project.

#### Option 1: Run the Interactive Web App (Recommended)

This is the best way to see the project in action.

```bash
python app.py

```

This will launch a Flask web application. Access the interface in your browser at: http://localhost:5000/simple

Option 2: Run the Terminal Demo
This runs a full comparison directly in your terminal and opens Matplotlib windows for the charts.

```bash
python main_demo.py