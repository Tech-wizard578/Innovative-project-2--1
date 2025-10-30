"""
smart_scheduler.py - AI-Powered Process Scheduler
Place this file in: D:\SmartSched\smart_scheduler.py
"""

import numpy as np
from collections import deque
import sys
import os

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
            self.predictor = BurstTimePredictor()
            if not self.predictor.load_model('models/'):
                print("⚙️  Training new ML model...")
                try:
                    self.predictor.train()
                except:
                    print("⚠️  ML training failed. Running without ML.")
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
                
                print(f"   P{proc.pid}: Actual={proc.burst_time}, "
                      f"Predicted={predicted}, Error={proc.prediction_error}")
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
                time += 1
                continue
            
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
                time += 1
                continue
            
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
            algorithms[self.algorithm]()
        else:
            print(f"❌ Unknown algorithm: {self.algorithm}")
            return None
        
        return self.get_metrics()
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if not self.completed:
            return None
        
        avg_waiting = np.mean([p.waiting_time for p in self.completed])
        avg_turnaround = np.mean([p.turnaround_time for p in self.completed])
        avg_response = np.mean([p.response_time for p in self.completed])
        
        total_burst = sum([p.original_burst for p in self.completed])
        total_time = max([p.completion_time for p in self.completed])
        cpu_utilization = (total_burst / total_time) * 100 if total_time > 0 else 0
        
        throughput = len(self.completed) / total_time if total_time > 0 else 0
        
        return {
            'avg_waiting_time': avg_waiting,
            'avg_turnaround_time': avg_turnaround,
            'avg_response_time': avg_response,
            'cpu_utilization': cpu_utilization,
            'throughput': throughput,
            'context_switches': self.context_switches
        }
    
    def print_results(self):
        """Display results"""
        print("\n" + "="*75)
        print("📊 SCHEDULING RESULTS")
        print("="*75)
        
        print(f"\n{'PID':<6} {'Arrival':<10} {'Burst':<8} {'Pred':<8} "
              f"{'Comp':<8} {'TAT':<8} {'WT':<8} {'RT':<8}")
        print("-"*75)
        
        for p in self.completed:
            pred_str = str(p.predicted_burst) if p.predicted_burst else "N/A"
            print(f"{p.pid:<6} {p.arrival_time:<10} {p.original_burst:<8} "
                  f"{pred_str:<8} {p.completion_time:<8} "
                  f"{p.turnaround_time:<8} {p.waiting_time:<8} {p.response_time:<8}")
        
        metrics = self.get_metrics()
        print("\n" + "="*75)
        print("📈 PERFORMANCE METRICS")
        print("="*75)
        print(f"Average Waiting Time:     {metrics['avg_waiting_time']:.2f} units")
        print(f"Average Turnaround Time:  {metrics['avg_turnaround_time']:.2f} units")
        print(f"Average Response Time:    {metrics['avg_response_time']:.2f} units")
        print(f"CPU Utilization:          {metrics['cpu_utilization']:.2f}%")
        print(f"Throughput:               {metrics['throughput']:.4f} processes/unit")
        print(f"Context Switches:         {metrics['context_switches']}")
        print("="*75)