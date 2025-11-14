"""
smart_scheduler.py - AI-Powered Process Scheduler
Place this file in: D:\SmartSched\smart_scheduler.py
"""

import numpy as np
from collections import deque
import sys
import os
import random
import time

# Import ML predictor
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
    from burst_predictor import BurstTimePredictor
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML predictor not available: {e}")


class Process:
    """Represents a process in the system"""
    def __init__(self, pid, arrival_time, burst_time, priority=5,
                 process_size=100, process_type=0, memory_usage=64, cpu_affinity=0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.original_burst = burst_time
        self.remaining_time = burst_time
        self.priority = priority
        self.process_size = process_size
        self.process_type = process_type
        self.memory_usage = memory_usage
        self.cpu_affinity = cpu_affinity

        # Scheduling metrics
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = -1
        self.start_time = -1

        # ML prediction
        self.predicted_burst = None
        self.prediction_error = 0

    def __repr__(self):
        return f"P{self.pid}(AT={self.arrival_time}, BT={self.burst_time}, Pri={self.priority})"


class EnergyAwareScheduler:
    def __init__(self, base_scheduler):
        self.scheduler = base_scheduler
        self.power_states = {
            'active': 1.0,
            'turbo': 1.5,
            'powersave': 0.6
        }
        self.energy_budget = 100.0  # Energy units

    def calculate_energy_consumption(self, schedule):
        """Calculate total energy consumption for a schedule"""
        total_energy = 0
        energy_history = []

        if not schedule:
            return {
                'total_energy': 0.0,
                'energy_history': [],
                'co2_emissions': 0.0,
                'cost_savings': 0.0
            }

        for entry in schedule:
            # Simulate CPU load based on process characteristics
            # tolerate missing keys
            cpu_usage = entry.get('cpu_usage', None)
            if cpu_usage is None:
                # approximate: map duration to load if possible
                duration = max(0, entry.get('end', 0) - entry.get('start', 0))
                cpu_load = min(1.0, duration / (duration + 10)) if duration > 0 else 0.5
            else:
                cpu_load = min(1.0, cpu_usage / 100.0)

            # Determine power state based on load
            power_state = self.determine_power_state(cpu_load)
            energy_consumed = cpu_load * self.power_states[power_state]

            total_energy += energy_consumed
            energy_history.append({
                'time': entry.get('start', 0),
                'energy': round(energy_consumed, 4),
                'power_state': power_state,
                'cpu_load': round(cpu_load, 4)
            })

        total_energy = round(total_energy, 4)
        return {
            'total_energy': total_energy,
            'energy_history': energy_history,
            'co2_emissions': round(total_energy * 0.0004, 6),  # kg CO2
            'cost_savings': round((total_energy / 100) * 0.12, 6)  # $ at $0.12/kWh
        }

    def determine_power_state(self, cpu_load):
        """Determine optimal power state based on CPU load"""
        if cpu_load > 0.8:
            return 'turbo'
        elif cpu_load < 0.3:
            return 'powersave'
        else:
            return 'active'

    def optimize_for_energy(self, processes):
        """Optimize scheduling for energy efficiency"""
        # Run normal scheduling first
        metrics = self.scheduler.run()  # run() already safe-guards returning a dict

        # Get the schedule
        schedule = self.scheduler.gantt_chart or []

        # Calculate energy consumption
        energy_metrics = self.calculate_energy_consumption(schedule)

        # Ensure metrics is a dict
        if metrics is None:
            metrics = {}

        # Add energy metrics to regular metrics
        metrics.update({
            'energy_consumption': energy_metrics.get('total_energy', 0.0),
            'co2_emissions': energy_metrics.get('co2_emissions', 0.0),
            'cost_savings': energy_metrics.get('cost_savings', 0.0),
            'energy_efficiency': self.calculate_energy_efficiency(metrics)
        })

        return metrics

    def calculate_energy_efficiency(self, metrics):
        """Calculate energy efficiency score"""
        # Simple efficiency calculation
        cpu_util = metrics.get('cpu_utilization', 50)
        energy = metrics.get('energy_consumption', 50)

        # Higher efficiency = high CPU utilization with low energy
        try:
            efficiency = (cpu_util / energy) * 10 if energy > 0 else 0
        except Exception:
            efficiency = 0
        return round(min(100, efficiency), 2)


class ExplainableScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def explain_decision(self, process, selected_algo):
        """Generate human-readable explanation for scheduling decision"""

        # Feature importance analysis
        features = {
            'burst_time': process.burst_time,
            'priority': process.priority,
            'process_size': process.process_size,
            'process_type': process.process_type,
            'arrival_time': process.arrival_time
        }

        # Calculate feature importance (simplified)
        shap_values = self.calculate_shap_importance(features, selected_algo)

        # Generate reasoning based on algorithm
        reasoning = self.generate_reasoning(features, selected_algo)

        # Get alternative algorithms
        alternatives = self.get_alternative_algorithms(features)

        explanation = {
            'algorithm': selected_algo,
            'confidence': float(self.calculate_confidence(selected_algo, features)),
            'key_factors': shap_values,
            'reasoning': reasoning,
            'alternative_algorithms': alternatives,
            'process_characteristics': {
                'type': self.get_process_type_name(process.process_type),
                'size_category': self.categorize_process_size(process.process_size),
                'priority_level': self.categorize_priority(process.priority)
            }
        }

        return explanation

    def calculate_shap_importance(self, features, algorithm):
        """Calculate simplified SHAP-like feature importance"""
        importance = {}

        if algorithm == 'SJF' or algorithm == 'SRTF':
            importance = {
                'burst_time': 0.8,
                'priority': 0.1,
                'process_size': 0.05,
                'arrival_time': 0.05
            }
        elif algorithm == 'PRIORITY':
            importance = {
                'priority': 0.7,
                'burst_time': 0.2,
                'arrival_time': 0.1
            }
        elif algorithm == 'RR':
            importance = {
                'arrival_time': 0.4,
                'burst_time': 0.3,
                'priority': 0.3
            }
        else:  # SMART_HYBRID or FCFS
            importance = {
                'burst_time': 0.4,
                'priority': 0.3,
                'arrival_time': 0.2,
                'process_size': 0.1
            }

        return importance

    def generate_reasoning(self, features, algorithm):
        """Generate human-readable reasoning"""
        reasons = []

        if algorithm == 'SJF':
            reasons.append(f"Selected SJF because process has burst time of {features['burst_time']} units, which is optimal for shortest job first scheduling.")
        elif algorithm == 'PRIORITY':
            reasons.append(f"Selected Priority scheduling because process has priority level {features['priority']}.")
        elif algorithm == 'RR':
            reasons.append("Selected Round Robin for fair time-sharing among processes.")
        elif algorithm == 'SMART_HYBRID':
            reasons.append("Selected Smart Hybrid algorithm which uses ML predictions and adaptive quantum.")
        else:
            reasons.append(f"Selected {algorithm} based on workload characteristics analysis.")

        # Add feature-based reasoning
        if features['burst_time'] < 10:
            reasons.append("Process has short burst time, making it suitable for quick execution.")
        elif features['burst_time'] > 50:
            reasons.append("Process has long burst time, requiring careful scheduling consideration.")

        if features['priority'] > 7:
            reasons.append("Process has high priority and should be executed promptly.")

        return " ".join(reasons)

    def get_alternative_algorithms(self, features):
        """Suggest alternative algorithms"""
        alternatives = []

        # Based on process characteristics
        if features['burst_time'] < 15:
            alternatives.append('SJF')
        if features['priority'] > 5:
            alternatives.append('PRIORITY')
        if features['process_type'] == 3:  # Interactive
            alternatives.append('RR')

        # Always suggest SMART_HYBRID as best option
        if 'SMART_HYBRID' not in alternatives:
            alternatives.append('SMART_HYBRID')

        return list(set(alternatives))

    def calculate_confidence(self, algorithm, features):
        """Calculate confidence score for algorithm selection"""
        base_confidence = 0.8

        # Adjust based on features
        if algorithm == 'SJF' and features['burst_time'] < 20:
            base_confidence += 0.1
        elif algorithm == 'PRIORITY' and features['priority'] > 7:
            base_confidence += 0.15
        elif algorithm == 'RR' and features['process_type'] == 3:
            base_confidence += 0.12

        return min(0.95, base_confidence)

    def get_process_type_name(self, process_type):
        """Convert process type number to name"""
        types = {0: 'CPU-Bound', 1: 'I/O-Bound', 2: 'Mixed', 3: 'Interactive'}
        return types.get(process_type, 'Unknown')

    def categorize_process_size(self, size):
        """Categorize process size"""
        if size < 200:
            return 'Small'
        elif size < 600:
            return 'Medium'
        else:
            return 'Large'

    def categorize_priority(self, priority):
        """Categorize priority level"""
        if priority <= 3:
            return 'Low'
        elif priority <= 7:
            return 'Medium'
        else:
            return 'High'


class SmartScheduler:
    """AI-Powered Process Scheduler"""

    def __init__(self, use_ml=True, quantum=4, algorithm='SMART_HYBRID'):
        self.processes = []
        self.completed = []
        self.current_time = 0
        self.quantum = quantum
        self.use_ml = use_ml and ML_AVAILABLE
        self.algorithm = algorithm
        self.gantt_chart = []
        self.context_switches = 0

        # Initialize ML predictor
        if self.use_ml:
            try:
                self.predictor = BurstTimePredictor()
                if not self.predictor.load_model('models/'):
                    print("⚙️  Training new ML model...")
                    try:
                        self.predictor.train()
                    except Exception:
                        print("⚠️  ML training failed. Running without ML.")
                        self.use_ml = False
            except Exception as e:
                print(f"⚠️  ML predictor init failed: {e}")
                self.use_ml = False

    def add_process(self, process):
        """Add a single process"""
        self.processes.append(process)

    def add_processes_batch(self, processes_list):
        """Add multiple processes"""
        self.processes.extend(processes_list)

    def reset(self):
        """Reset scheduler state"""
        self.completed = []
        self.current_time = 0
        self.gantt_chart = []
        self.context_switches = 0
        for proc in self.processes:
            proc.remaining_time = proc.original_burst
            proc.completion_time = 0
            proc.waiting_time = 0
            proc.turnaround_time = 0
            proc.response_time = -1
            proc.start_time = -1
            proc.predicted_burst = None
            proc.prediction_error = 0

    def predict_burst_times(self):
        """Use ML to predict burst times"""
        if not self.use_ml:
            return

        print("\n🔮 Predicting burst times using ML...")

        for proc in self.processes:
            features = {
                'process_size': proc.process_size,
                'priority': proc.priority,
                'arrival_time': proc.arrival_time,
                'prev_burst_avg': 20,
                'process_type': proc.process_type,
                'time_of_day': self.current_time % 24,
                'memory_usage': proc.memory_usage,
                'cpu_affinity': proc.cpu_affinity
            }

            try:
                predicted = self.predictor.predict(features)
                proc.predicted_burst = predicted
                proc.prediction_error = abs(proc.burst_time - predicted)
                print(f"   P{proc.pid}: Actual={proc.burst_time}, Predicted={predicted}, Error={proc.prediction_error}")
            except Exception as e:
                print(f"   ⚠️  Prediction failed for P{proc.pid}: {e}")
                proc.predicted_burst = proc.burst_time

    def schedule_fcfs(self):
        """First Come First Serve"""
        processes = sorted(self.processes, key=lambda x: x.arrival_time)
        time = 0

        for proc in processes:
            if time < proc.arrival_time:
                time = proc.arrival_time

            proc.start_time = time
            proc.response_time = time - proc.arrival_time

            self.gantt_chart.append({
                'pid': proc.pid,
                'start': time,
                'end': time + proc.burst_time
            })

            time += proc.burst_time
            proc.completion_time = time
            proc.turnaround_time = time - proc.arrival_time
            proc.waiting_time = proc.turnaround_time - proc.burst_time

        self.completed = processes

    def schedule_sjf(self):
        """Shortest Job First"""
        time = 0
        ready_queue = []
        remaining = sorted(self.processes, key=lambda x: x.arrival_time).copy()
        completed = []

        while remaining or ready_queue:
            while remaining and remaining[0].arrival_time <= time:
                ready_queue.append(remaining.pop(0))

            if not ready_queue:
                time = remaining[0].arrival_time if remaining else time
                continue

            if self.use_ml:
                ready_queue.sort(key=lambda p: p.predicted_burst if p.predicted_burst else p.burst_time)
            else:
                ready_queue.sort(key=lambda p: p.burst_time)

            current = ready_queue.pop(0)

            if current.start_time == -1:
                current.start_time = time
                current.response_time = time - current.arrival_time

            self.gantt_chart.append({
                'pid': current.pid,
                'start': time,
                'end': time + current.burst_time
            })

            time += current.burst_time
            current.completion_time = time
            current.turnaround_time = time - current.arrival_time
            current.waiting_time = current.turnaround_time - current.burst_time
            completed.append(current)

        self.completed = completed

    def schedule_srtf(self):
        """Shortest Remaining Time First"""
        time = 0
        ready_queue = []
        remaining = sorted(self.processes, key=lambda x: x.arrival_time).copy()
        completed = []
        current = None

        while remaining or ready_queue or current:
            while remaining and remaining[0].arrival_time <= time:
                ready_queue.append(remaining.pop(0))

            if current and current.remaining_time > 0:
                ready_queue.append(current)

            if not ready_queue:
                if remaining:
                    time = remaining[0].arrival_time
                else:
                    break
                continue

            ready_queue.sort(key=lambda p: p.remaining_time)

            prev_current = current
            current = ready_queue.pop(0)

            if current != prev_current and prev_current:
                self.context_switches += 1

            if current.start_time == -1:
                current.start_time = time
                current.response_time = time - current.arrival_time

            exec_time = 1

            if not self.gantt_chart or self.gantt_chart[-1]['pid'] != current.pid:
                self.gantt_chart.append({
                    'pid': current.pid,
                    'start': time,
                    'end': time + exec_time
                })
            else:
                self.gantt_chart[-1]['end'] += exec_time

            time += exec_time
            current.remaining_time -= exec_time

            if current.remaining_time == 0:
                current.completion_time = time
                current.turnaround_time = time - current.arrival_time
                current.waiting_time = current.turnaround_time - current.original_burst
                completed.append(current)
                current = None

        self.completed = completed

    def schedule_priority(self):
        """Priority Scheduling"""
        time = 0
        ready_queue = []
        remaining = sorted(self.processes, key=lambda x: x.arrival_time).copy()
        completed = []

        while remaining or ready_queue:
            while remaining and remaining[0].arrival_time <= time:
                ready_queue.append(remaining.pop(0))

            if not ready_queue:
                time = remaining[0].arrival_time if remaining else time
                continue

            ready_queue.sort(key=lambda p: (p.priority, p.arrival_time))

            current = ready_queue.pop(0)

            if current.start_time == -1:
                current.start_time = time
                current.response_time = time - current.arrival_time

            self.gantt_chart.append({
                'pid': current.pid,
                'start': time,
                'end': time + current.burst_time
            })

            time += current.burst_time
            current.completion_time = time
            current.turnaround_time = time - current.arrival_time
            current.waiting_time = current.turnaround_time - current.burst_time
            completed.append(current)

        self.completed = completed

    def schedule_round_robin(self):
        """Round Robin Scheduling"""
        time = 0
        ready_queue = deque()
        remaining = sorted(self.processes, key=lambda x: x.arrival_time).copy()
        completed = []

        while remaining or ready_queue:
            while remaining and remaining[0].arrival_time <= time:
                proc = remaining.pop(0)
                ready_queue.append(proc)

            if not ready_queue:
                # if there are still future arrivals, fast-forward
                if remaining:
                    time = remaining[0].arrival_time
                    continue
                else:
                    break

            current = ready_queue.popleft()

            if current.start_time == -1:
                current.start_time = time
                current.response_time = time - current.arrival_time

            exec_time = min(self.quantum, current.remaining_time)

            self.gantt_chart.append({
                'pid': current.pid,
                'start': time,
                'end': time + exec_time
            })

            time += exec_time
            current.remaining_time -= exec_time

            while remaining and remaining[0].arrival_time <= time:
                proc = remaining.pop(0)
                ready_queue.append(proc)

            if current.remaining_time == 0:
                current.completion_time = time
                current.turnaround_time = time - current.arrival_time
                current.waiting_time = current.turnaround_time - current.original_burst
                completed.append(current)
            else:
                ready_queue.append(current)
                self.context_switches += 1

        self.completed = completed

    def schedule_smart_hybrid(self):
        """SmartSched Hybrid Algorithm"""
        if self.use_ml:
            self.predict_burst_times()

        ready_queue = []
        time = 0
        completed = []
        remaining = sorted(self.processes, key=lambda x: x.arrival_time).copy()

        while remaining or ready_queue:
            while remaining and remaining[0].arrival_time <= time:
                proc = remaining.pop(0)
                ready_queue.append(proc)

            if not ready_queue:
                if remaining:
                    time = remaining[0].arrival_time
                    continue
                else:
                    break

            # Smart selection
            ready_queue.sort(key=lambda p: (
                p.predicted_burst if p.predicted_burst else p.remaining_time,
                -p.priority,
                p.arrival_time
            ))

            current = ready_queue.pop(0)

            if current.start_time == -1:
                current.start_time = time
                current.response_time = time - current.arrival_time

            # Adaptive quantum
            if current.process_type == 3:
                exec_time = min(2, current.remaining_time)
            else:
                exec_time = min(self.quantum, current.remaining_time)

            self.gantt_chart.append({
                'pid': current.pid,
                'start': time,
                'end': time + exec_time
            })

            time += exec_time
            current.remaining_time -= exec_time

            if current.remaining_time == 0:
                current.completion_time = time
                current.turnaround_time = time - current.arrival_time
                current.waiting_time = current.turnaround_time - current.original_burst
                completed.append(current)
            else:
                current.priority = min(10, current.priority + 1)
                ready_queue.append(current)
                self.context_switches += 1

        self.completed = completed

    def run(self, algorithm=None):
        """Execute scheduling algorithm"""
        if algorithm:
            self.algorithm = algorithm

        print(f"\n🚀 Running {self.algorithm} Scheduling...")

        self.reset()

        algorithms = {
            'FCFS': self.schedule_fcfs,
            'SJF': self.schedule_sjf,
            'SRTF': self.schedule_srtf,
            'PRIORITY': self.schedule_priority,
            'RR': self.schedule_round_robin,
            'SMART_HYBRID': self.schedule_smart_hybrid
        }

        if self.algorithm in algorithms:
            try:
                algorithms[self.algorithm]()
            except Exception as e:
                print(f"❌ Error during scheduling ({self.algorithm}): {e}")
        else:
            print(f"❌ Unknown algorithm: {self.algorithm}")

        # Always return a metrics dict — never None
        metrics = self.get_metrics()
        if metrics is None:
            # Fallback metrics if scheduling produced no completed entries
            total_burst = sum([p.original_burst for p in self.processes]) if self.processes else 0
            total_time = 0
            if self.gantt_chart:
                total_time = max([entry.get('end', 0) for entry in self.gantt_chart])
            cpu_utilization = (total_burst / total_time) * 100 if total_time > 0 else 0
            throughput = len(self.completed) / total_time if total_time > 0 else 0

            metrics = {
                'avg_waiting_time': 0.0,
                'avg_turnaround_time': 0.0,
                'avg_response_time': 0.0,
                'cpu_utilization': cpu_utilization,
                'throughput': throughput,
                'context_switches': self.context_switches
            }

        # attach gantt chart for callers that expect it
        metrics['gantt_chart'] = self.gantt_chart
        return metrics

    def get_metrics(self):
        """Calculate performance metrics"""
        # prefer completed list, but fall back to any processes with completion_time set
        completed_list = self.completed if self.completed else [p for p in self.processes if getattr(p, 'completion_time', 0) > 0]

        if not completed_list:
            return None

        try:
            avg_waiting = float(np.mean([p.waiting_time for p in completed_list]))
            avg_turnaround = float(np.mean([p.turnaround_time for p in completed_list]))
            avg_response = float(np.mean([p.response_time for p in completed_list if p.response_time >= 0])) if any(p.response_time >= 0 for p in completed_list) else 0.0

            total_burst = float(sum([p.original_burst for p in completed_list]))
            total_time = float(max([p.completion_time for p in completed_list])) if completed_list else 0.0
            cpu_utilization = (total_burst / total_time) * 100 if total_time > 0 else 0.0

            throughput = len(completed_list) / total_time if total_time > 0 else 0.0

            return {
                'avg_waiting_time': round(avg_waiting, 4),
                'avg_turnaround_time': round(avg_turnaround, 4),
                'avg_response_time': round(avg_response, 4),
                'cpu_utilization': round(cpu_utilization, 4),
                'throughput': round(throughput, 6),
                'context_switches': int(self.context_switches)
            }
        except Exception as e:
            print(f"❌ Error computing metrics: {e}")
            return None

    def get_energy_aware_metrics(self):
        """Get energy-aware performance metrics"""
        metrics = self.get_metrics()

        if metrics is None:
            # safe fallback
            metrics = {
                'avg_waiting_time': 0.0,
                'avg_turnaround_time': 0.0,
                'avg_response_time': 0.0,
                'cpu_utilization': 0.0,
                'throughput': 0.0,
                'context_switches': int(self.context_switches)
            }

        # Add energy awareness
        energy_aware = EnergyAwareScheduler(self)
        energy_metrics = energy_aware.calculate_energy_consumption(self.gantt_chart)

        metrics.update({
            'energy_consumption': energy_metrics.get('total_energy', 0.0),
            'co2_emissions': energy_metrics.get('co2_emissions', 0.0),
            'cost_savings': energy_metrics.get('cost_savings', 0.0),
            'energy_efficiency': energy_aware.calculate_energy_efficiency(metrics)
        })

        return metrics

    def print_results(self):
        """Display results"""
        print("\n" + "=" * 75)
        print("📊 SCHEDULING RESULTS")
        print("=" * 75)

        if not self.completed:
            print("No completed processes to display.")
        else:
            print(f"\n{'PID':<6} {'Arrival':<10} {'Burst':<8} {'Pred':<8} "
                  f"{'Comp':<8} {'TAT':<8} {'WT':<8} {'RT':<8}")
            print("-" * 75)

            for p in self.completed:
                pred_str = str(p.predicted_burst) if p.predicted_burst else "N/A"
                print(f"{p.pid:<6} {p.arrival_time:<10} {p.original_burst:<8} "
                      f"{pred_str:<8} {p.completion_time:<8} "
                      f"{p.turnaround_time:<8} {p.waiting_time:<8} {p.response_time:<8}")

        metrics = self.get_metrics()
        if not metrics:
            print("\nNo metrics available.")
            return

        print("\n" + "=" * 75)
        print("📈 PERFORMANCE METRICS")
        print("=" * 75)
        print(f"Average Waiting Time:     {metrics['avg_waiting_time']:.2f} units")
        print(f"Average Turnaround Time:  {metrics['avg_turnaround_time']:.2f} units")
        print(f"Average Response Time:    {metrics['avg_response_time']:.2f} units")
        print(f"CPU Utilization:          {metrics['cpu_utilization']:.2f}%")
        print(f"Throughput:               {metrics['throughput']:.6f} processes/unit")
        print(f"Context Switches:         {metrics['context_switches']}")
        print("=" * 75)
