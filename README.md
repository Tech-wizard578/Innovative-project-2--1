# 🚀 SmartSched: An AI-Powered Process Scheduler

SmartSched is an advanced process scheduling simulator that leverages machine learning and reinforcement learning to optimize scheduling decisions. It is designed to compare traditional scheduling algorithms (like FCFS and Priority RR) against a "smart" hybrid algorithm that uses an ML model to predict process burst times.

Furthermore, it includes a Q-learning agent that analyzes the incoming workload and dynamically selects the best scheduling algorithm (FCFS, Priority, or ML-hybrid) to minimize metrics like Average Waiting Time.

This project includes:
* An interactive **Streamlit Web App** (`app.py`) for visual comparison.
* A terminal-based demo (`main_demo.py`).
* A full training pipeline (`train_models.py`) that learns from real-world **Google Borg** cluster data.

---

## Core Features

* **ML Burst Time Prediction:** Uses Random Forest and LSTM (PyTorch) models to predict a process's burst time based on its features.
* **Data-Driven Models:** The ML predictors are trained on a processed dataset from **Google's Borg cluster**, allowing them to learn realistic process behavior.
* **Reinforcement Learning Agent:** A Q-learning agent is trained to analyze workload characteristics (e.g., 'CPU-Bound', 'Interactive', queue length) and select the optimal scheduling policy.
* **Advanced Schedulers:**
    * **FCFS:** Standard non-preemptive First-Come-First-Served.
    * **Priority RR:** A fully **preemptive** Priority Round Robin scheduler.
    * **SMART\_HYBRID (PSPJF):** A fully **preemptive** scheduler (Preemptive Shortest *Predicted* Job First) that uses the ML model's predictions to prioritize jobs.
* **Interactive Web App:** A rich Streamlit app (`app.py`) to run simulations, compare all algorithms side-by-side, and visualize results.
* **Comprehensive Visualization:** Generates Plotly Gantt charts, performance metric comparisons (Avg. Wait/Turnaround/Response Time), and ML prediction accuracy plots.

---

## ⚙️ Project Architecture

The project follows a clear Data -> Train -> Simulate pipeline.

1.  **Data Processing (`preprocess_borg_data.py`)**
    * This script (which you must run once) takes raw Google Borg data (not included in the repo) and processes it.
    * It cleans, samples, and feature-engineers the data, saving the result as `data/processed_borg_data_v1.csv`.

2.  **Model Training (`train_models.py`)**
    * This script loads the processed `data/processed_borg_data_v1.csv`.
    * It trains the **Random Forest** (`trained_model.pkl`) and **LSTM** (`lstm_model.pth`, `lstm_scaler.pkl`) predictors on this real-world data.
    * It then trains the **Q-learning Agent** (`rl_agent_q_table.pkl`) by running thousands of simulations using the (just-trained) RF predictor as part of its environment.

3.  **Simulation & Application (`app.py` / `main_demo.py`)**
    * The `SmartScheduler` class (`schedulers/smart_scheduler.py`) is the core engine. On initialization, it pre-loads the trained ML models.
    * The Streamlit App (`app.py`) or Terminal Demo (`main_demo.py`) creates a synthetic workload.
    * It then runs this workload through each of the scheduler's algorithms (FCFS, Priority RR, SMART\_HYBRID, and the RL Agent's choice).
    * Finally, it uses `visualization/visualization.py` to plot the results and metrics from each run.

---

## 🔧 Technology Stack

* **Simulation:** Python 3.x
* **Web App:** `streamlit`
* **Data & ML:** `pandas`, `numpy`, `scikit-learn` (for Random Forest & Scaler)
* **Deep Learning:** `torch` (for LSTM)
* **Visualization:** `plotly`, `matplotlib`, `seaborn`

*(See `requirements.txt` for a full list of dependencies)*

---

## 🚀 How to Run

### Step 1: Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/innovative-project-2--1.git](https://github.com/Tech-wizard578/innovative-project-2--1.git)
    cd innovative-project-2--1
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data & Model Training (One-Time Setup)

You must process the data (if you have the raw files) and train the models before you can run the simulations.

1.  **Process the Data (Optional)**
    *(This project already provides a pre-processed data file: `data/processed_borg_data_v1.csv`. If you had the raw Google Borg trace data, you would run this first.)*
    ```bash
    # (Optional) python preprocess_borg_data.py
    ```

2.  **Train All Models**
    This is the most important step. This script will train the RF, LSTM, and RL Agent models using the data in the `/data` folder.
    ```bash
    python train_models.py
    ```
    This will create the following essential files in the `models/` directory:
    * `trained_model.pkl` (Random Forest)
    * `lstm_model.pth` (LSTM Model)
    * `lstm_scaler.pkl` (LSTM Scaler)
    * `rl_agent_q_table.pkl` (RL Agent)

### Step 3: Run the Application

You have two ways to run the project.

#### Option 1: Run the Interactive Web App (Recommended)

This is the best way to see the project in action.

```bash
streamlit run app.py

This will launch a web application in your browser. You can select different workload types, run the simulation, and see a full comparison of all schedulers with interactive charts.

Markdown

# 🚀 SmartSched: An AI-Powered Process Scheduler

SmartSched is an advanced process scheduling simulator that leverages machine learning and reinforcement learning to optimize scheduling decisions. It is designed to compare traditional scheduling algorithms (like FCFS and Priority RR) against a "smart" hybrid algorithm that uses an ML model to predict process burst times.

Furthermore, it includes a Q-learning agent that analyzes the incoming workload and dynamically selects the best scheduling algorithm (FCFS, Priority, or ML-hybrid) to minimize metrics like Average Waiting Time.

This project includes:
* An interactive **Streamlit Web App** (`app.py`) for visual comparison.
* A terminal-based demo (`main_demo.py`).
* A full training pipeline (`train_models.py`) that learns from real-world **Google Borg** cluster data.

---

## Core Features

* **ML Burst Time Prediction:** Uses Random Forest and LSTM (PyTorch) models to predict a process's burst time based on its features.
* **Data-Driven Models:** The ML predictors are trained on a processed dataset from **Google's Borg cluster**, allowing them to learn realistic process behavior.
* **Reinforcement Learning Agent:** A Q-learning agent is trained to analyze workload characteristics (e.g., 'CPU-Bound', 'Interactive', queue length) and select the optimal scheduling policy.
* **Advanced Schedulers:**
    * **FCFS:** Standard non-preemptive First-Come-First-Served.
    * **Priority RR:** A fully **preemptive** Priority Round Robin scheduler.
    * **SMART\_HYBRID (PSPJF):** A fully **preemptive** scheduler (Preemptive Shortest *Predicted* Job First) that uses the ML model's predictions to prioritize jobs.
* **Interactive Web App:** A rich Streamlit app (`app.py`) to run simulations, compare all algorithms side-by-side, and visualize results.
* **Comprehensive Visualization:** Generates Plotly Gantt charts, performance metric comparisons (Avg. Wait/Turnaround/Response Time), and ML prediction accuracy plots.

---

## ⚙️ Project Architecture

The project follows a clear Data -> Train -> Simulate pipeline.

1.  **Data Processing (`preprocess_borg_data.py`)**
    * This script (which you must run once) takes raw Google Borg data (not included in the repo) and processes it.
    * It cleans, samples, and feature-engineers the data, saving the result as `data/processed_borg_data_v1.csv`.

2.  **Model Training (`train_models.py`)**
    * This script loads the processed `data/processed_borg_data_v1.csv`.
    * It trains the **Random Forest** (`trained_model.pkl`) and **LSTM** (`lstm_model.pth`, `lstm_scaler.pkl`) predictors on this real-world data.
    * It then trains the **Q-learning Agent** (`rl_agent_q_table.pkl`) by running thousands of simulations using the (just-trained) RF predictor as part of its environment.

3.  **Simulation & Application (`app.py` / `main_demo.py`)**
    * The `SmartScheduler` class (`schedulers/smart_scheduler.py`) is the core engine. On initialization, it pre-loads the trained ML models.
    * The Streamlit App (`app.py`) or Terminal Demo (`main_demo.py`) creates a synthetic workload.
    * It then runs this workload through each of the scheduler's algorithms (FCFS, Priority RR, SMART\_HYBRID, and the RL Agent's choice).
    * Finally, it uses `visualization/visualization.py` to plot the results and metrics from each run.

---

## 🔧 Technology Stack

* **Simulation:** Python 3.x
* **Web App:** `streamlit`
* **Data & ML:** `pandas`, `numpy`, `scikit-learn` (for Random Forest & Scaler)
* **Deep Learning:** `torch` (for LSTM)
* **Visualization:** `plotly`, `matplotlib`, `seaborn`

*(See `requirements.txt` for a full list of dependencies)*

---

## 🚀 How to Run

### Step 1: Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/innovative-project-2--1.git](https://github.com/your-username/innovative-project-2--1.git)
    cd innovative-project-2--1
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data & Model Training (One-Time Setup)

You must process the data (if you have the raw files) and train the models before you can run the simulations.

1.  **Process the Data (Optional)**
    *(This project already provides a pre-processed data file: `data/processed_borg_data_v1.csv`. If you had the raw Google Borg trace data, you would run this first.)*
    ```bash
    # (Optional) python preprocess_borg_data.py
    ```

2.  **Train All Models**
    This is the most important step. This script will train the RF, LSTM, and RL Agent models using the data in the `/data` folder.
    ```bash
    python train_models.py
    ```
    This will create the following essential files in the `models/` directory:
    * `trained_model.pkl` (Random Forest)
    * `lstm_model.pth` (LSTM Model)
    * `lstm_scaler.pkl` (LSTM Scaler)
    * `rl_agent_q_table.pkl` (RL Agent)

### Step 3: Run the Application

You have two ways to run the project.

#### Option 1: Run the Interactive Web App (Recommended)

This is the best way to see the project in action.

```bash
streamlit run app.py

This will launch a web application in your browser. You can select different workload types, run the simulation, and see a full comparison of all schedulers with interactive charts.

Option 2: Run the Terminal Demo
This runs a full comparison directly in your terminal and opens Matplotlib windows for the charts

```bash
python main_demo.py


Markdown

# 🚀 SmartSched: An AI-Powered Process Scheduler

SmartSched is an advanced process scheduling simulator that leverages machine learning and reinforcement learning to optimize scheduling decisions. It is designed to compare traditional scheduling algorithms (like FCFS and Priority RR) against a "smart" hybrid algorithm that uses an ML model to predict process burst times.

Furthermore, it includes a Q-learning agent that analyzes the incoming workload and dynamically selects the best scheduling algorithm (FCFS, Priority, or ML-hybrid) to minimize metrics like Average Waiting Time.

This project includes:
* An interactive **Streamlit Web App** (`app.py`) for visual comparison.
* A terminal-based demo (`main_demo.py`).
* A full training pipeline (`train_models.py`) that learns from real-world **Google Borg** cluster data.

---

## Core Features

* **ML Burst Time Prediction:** Uses Random Forest and LSTM (PyTorch) models to predict a process's burst time based on its features.
* **Data-Driven Models:** The ML predictors are trained on a processed dataset from **Google's Borg cluster**, allowing them to learn realistic process behavior.
* **Reinforcement Learning Agent:** A Q-learning agent is trained to analyze workload characteristics (e.g., 'CPU-Bound', 'Interactive', queue length) and select the optimal scheduling policy.
* **Advanced Schedulers:**
    * **FCFS:** Standard non-preemptive First-Come-First-Served.
    * **Priority RR:** A fully **preemptive** Priority Round Robin scheduler.
    * **SMART\_HYBRID (PSPJF):** A fully **preemptive** scheduler (Preemptive Shortest *Predicted* Job First) that uses the ML model's predictions to prioritize jobs.
* **Interactive Web App:** A rich Streamlit app (`app.py`) to run simulations, compare all algorithms side-by-side, and visualize results.
* **Comprehensive Visualization:** Generates Plotly Gantt charts, performance metric comparisons (Avg. Wait/Turnaround/Response Time), and ML prediction accuracy plots.

---

## ⚙️ Project Architecture

The project follows a clear Data -> Train -> Simulate pipeline.

1.  **Data Processing (`preprocess_borg_data.py`)**
    * This script (which you must run once) takes raw Google Borg data (not included in the repo) and processes it.
    * It cleans, samples, and feature-engineers the data, saving the result as `data/processed_borg_data_v1.csv`.

2.  **Model Training (`train_models.py`)**
    * This script loads the processed `data/processed_borg_data_v1.csv`.
    * It trains the **Random Forest** (`trained_model.pkl`) and **LSTM** (`lstm_model.pth`, `lstm_scaler.pkl`) predictors on this real-world data.
    * It then trains the **Q-learning Agent** (`rl_agent_q_table.pkl`) by running thousands of simulations using the (just-trained) RF predictor as part of its environment.

3.  **Simulation & Application (`app.py` / `main_demo.py`)**
    * The `SmartScheduler` class (`schedulers/smart_scheduler.py`) is the core engine. On initialization, it pre-loads the trained ML models.
    * The Streamlit App (`app.py`) or Terminal Demo (`main_demo.py`) creates a synthetic workload.
    * It then runs this workload through each of the scheduler's algorithms (FCFS, Priority RR, SMART\_HYBRID, and the RL Agent's choice).
    * Finally, it uses `visualization/visualization.py` to plot the results and metrics from each run.

---

## 🔧 Technology Stack

* **Simulation:** Python 3.x
* **Web App:** `streamlit`
* **Data & ML:** `pandas`, `numpy`, `scikit-learn` (for Random Forest & Scaler)
* **Deep Learning:** `torch` (for LSTM)
* **Visualization:** `plotly`, `matplotlib`, `seaborn`

*(See `requirements.txt` for a full list of dependencies)*

---

## 🚀 How to Run

### Step 1: Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/innovative-project-2--1.git](https://github.com/your-username/innovative-project-2--1.git)
    cd innovative-project-2--1
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data & Model Training (One-Time Setup)

You must process the data (if you have the raw files) and train the models before you can run the simulations.

1.  **Process the Data (Optional)**
    *(This project already provides a pre-processed data file: `data/processed_borg_data_v1.csv`. If you had the raw Google Borg trace data, you would run this first.)*
    ```bash
    # (Optional) python preprocess_borg_data.py
    ```

2.  **Train All Models**
    This is the most important step. This script will train the RF, LSTM, and RL Agent models using the data in the `/data` folder.
    ```bash
    python train_models.py
    ```
    This will create the following essential files in the `models/` directory:
    * `trained_model.pkl` (Random Forest)
    * `lstm_model.pth` (LSTM Model)
    * `lstm_scaler.pkl` (LSTM Scaler)
    * `rl_agent_q_table.pkl` (RL Agent)

### Step 3: Run the Application

You have two ways to run the project.

#### Option 1: Run the Interactive Web App (Recommended)

This is the best way to see the project in action.

```bash
streamlit run app.py
This will launch a web application in your browser. You can select different workload types, run the simulation, and see a full comparison of all schedulers with interactive charts.

Option 2: Run the Terminal Demo
This runs a full comparison directly in your terminal and opens Matplotlib windows for the charts.

Bash

python main_demo.py
🔬 Component Deep-Dive
schedulers/smart_scheduler.py
This is the core engine of the project.

Process class: A data class to hold process information (PID, arrival time, burst time, etc.) and track its metrics (wait time, turnaround, etc.). Includes a reset_metrics() method.

SmartScheduler class:

__init__: Safely pre-loads the ML predictors (.pkl and .pth files) into memory. It does not train them, preventing app-hanging bugs.

add_processes_batch: Loads a list of processes and immediately calls predict_burst_times to get ML predictions for the whole batch.

predict_burst_times: Creates a pandas.DataFrame from the current processes (using features process_size, priority, process_type) and passes it to the loaded predictor (RF or LSTM) to get all predictions in one efficient call.

_run_preemptive_simulation: The corrected simulation engine. This generic function provides a true, event-driven preemptive simulation. It continuously checks for new arrivals that can interrupt the currently running process, ensuring correct preemptive logic.

schedule_fcfs: A simple non-preemptive algorithm.

schedule_priority_rr: Uses the preemptive engine (_run_preemptive_simulation) with a sort key of (p.priority, p.arrival_time).

schedule_smart_hybrid: This is Preemptive Shortest Predicted Job First (PSPJF). It uses the preemptive engine with a sort key of (p.predicted_burst, p.arrival_time).

analyze_workload: A crucial function for the RL agent. It takes a list of processes and discretizes the workload into a "state" (e.g., {'workload': 'cpu_bound', 'priority': 'high_variance', ...}).

models/
burst_predictor.py (Random Forest):

Defines a BurstTimePredictor class.

Trains a RandomForestRegressor model from scikit-learn.

The model learns to predict burst_time based on process_size, priority, and process_type.

lstm_predictor.py (LSTM):

Defines an LSTMPredictor class using PyTorch.

Trains an LSTM network on the same features: process_size, priority, and process_type.

It also saves a StandardScaler (lstm_scaler.pkl) used to normalize the data.

rl_agent.py (Q-Learning):

Defines a RLSchedulerAgent class.

Uses a dictionary (q_table) to store state-action values.

State: A string representation of the workload (e.g., 'workload=mixed;priority=low_variance;... ').

Action: One of the three algorithms: 'FCFS', 'PRIORITY_RR', or 'SMART_HYBRID'.

Reward: The negative average waiting time (-metrics['avg_waiting_time']). By trying to maximize this reward, the agent learns to minimize waiting time.