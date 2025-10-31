"""
multicore_scheduler.py - Multi-Core CPU Scheduler
Simulates parallel processing on multiple CPU cores with memory management
"""

import numpy as np
from collections import deque
from smart_scheduler import Process


class Core:
    """Represents a single CPU core"""
    def __init__(self, core_id):
        self.core_id = core_id
        self.current_process = None
        self.idle_time = 0
        self.execution_history = []
    
    def is_idle(self):
        return self.current_process is None
    
    def assign_process(self, process, current_time):
        """Assign a process to this core"""
        self.current_process = process
        if process.start_time == -1:
            process.start_time = current_time
            process.response_time = current_time - process.arrival_time
    
    def execute(self, time_slice, current_time):
        """Execute current process for time_slice"""
        if self.current_process is None:
            self.idle_time += time_slice
            return None
        
        exec_time = min(time_slice, self.current_process.remaining_time)
        
        self.execution_history.append({
            'pid': self.current_process.pid,
            'core': self.core_id,
            'start': current_time,
            'end': current_time + exec_time
        })
        
        self.current_process.remaining_time -= exec_time
        
        if self.current_process.remaining_time == 0:
            # Process completed
            completed = self.current_process
            completed.completion_time = current_time + exec_time
            completed.turnaround_time = completed.completion_time - completed.arrival_time
            completed.waiting_time = completed.turnaround_time - completed.original_burst
            self.current_process = None
            return completed
        
        return None


class MemoryManager:
    """Manages memory allocation for processes"""
    def __init__(self, total_memory=2048):  # MB
        self.total_memory = total_memory
        self.used_memory = 0
        self.allocated_processes = {}
    
    def can_allocate(self, process):
        """Check if there's enough memory for process"""
        return (self.used_memory + process.memory_usage) <= self.total_memory
    
    def allocate(self, process):
        """Allocate memory for process"""
        if self.can_allocate(process):
            self.used_memory += process.memory_usage
            self.allocated_processes[process.pid] = process.memory_usage
            return True
        return False
    
    def deallocate(self, process):
        """Free memory from completed process"""
        if process.pid in self.allocated_processes:
            self.used_memory -= self.allocated_processes[process.pid]
            del self.allocated_processes[process.pid]
    
    def get_memory_usage(self):
        """Get current memory utilization percentage"""
        return (self.used_memory / self.total_memory) * 100


class MultiCoreScheduler:
    """
    Multi-Core CPU Scheduler with Memory Management
    Supports parallel execution on multiple cores
    """
    
    def __init__(self, num_cores=4, total_memory=2048, use_ml=False, quantum=4):
        self.num_cores = num_cores
        self.cores = [Core(i) for i in range(num_cores)]
        self.memory_manager = MemoryManager(total_memory)
        self.use_ml = use_ml
        self.quantum = quantum
        
        self.processes = []
        self.ready_queue = deque()
        self.waiting_queue = deque()  # Processes waiting for memory
        self.completed = []
        self.gantt_chart = []
        self.context_switches = 0
        
        # ML predictor
        if self.use_ml:
            try:
                from burst_predictor import BurstTimePredictor
                self.predictor = BurstTimePredictor()
                if not self.predictor.load_model('models/'):
                    self.use_ml = False
            except:
                self.use_ml = False
    
    def add_processes(self, processes):
        """Add processes to scheduler"""
        self.processes = sorted(processes, key=lambda x: x.arrival_time)
    
    def predict_burst_times(self):
        """Use ML to predict burst times"""
        if not self.use_ml:
            return
        
        print("\n🔮 ML Predictions for Multi-Core Scheduling:")
        for proc in self.processes:
            features = {
                'process_size': proc.process_size,
                'priority': proc.priority,
                'arrival_time': proc.arrival_time,
                'prev_burst_avg': 20,
                'process_type': proc.process_type,
                'time_of_day': 14,
                'memory_usage': proc.memory_usage,
                'cpu_affinity': proc.cpu_affinity
            }
            
            try:
                pred = self.predictor.predict(features)
                proc.predicted_burst = pred
                print(f"   P{proc.pid}: Predicted={pred}, Actual={proc.burst_time}")
            except:
                proc.predicted_burst = proc.burst_time
    
    def select_best_core_for_process(self, process):
        """
        Select best core for process based on cpu_affinity
        Returns core_id or None if all busy
        """
        idle_cores = [c for c in self.cores if c.is_idle()]
        
        if not idle_cores:
            return None
        
        # Check CPU affinity
        if process.cpu_affinity < len(idle_cores):
            preferred_core = self.cores[process.cpu_affinity]
            if preferred_core.is_idle():
                return preferred_core.core_id
        
        # Return first idle core
        return idle_cores[0].core_id
    
    def schedule_multicore_rr(self):
        """
        Multi-core Round Robin with Memory Management
        """
        if self.use_ml:
            self.predict_burst_times()
        
        time = 0
        remaining = self.processes.copy()
        
        print(f"\n⚙️  Multi-Core Scheduling on {self.num_cores} cores")
        print(f"   Total Memory: {self.memory_manager.total_memory}MB")
        print(f"   Quantum: {self.quantum} units\n")
        
        while remaining or self.ready_queue or self.waiting_queue or any(not c.is_idle() for c in self.cores):
            # Add newly arrived processes to waiting queue (need memory allocation)
            while remaining and remaining[0].arrival_time <= time:
                proc = remaining.pop(0)
                self.waiting_queue.append(proc)
            
            # Try to allocate memory for waiting processes
            successfully_allocated = []
            for proc in list(self.waiting_queue):
                if self.memory_manager.allocate(proc):
                    self.ready_queue.append(proc)
                    successfully_allocated.append(proc)
            
            for proc in successfully_allocated:
                self.waiting_queue.remove(proc)
            
            # Assign processes to idle cores
            while self.ready_queue:
                proc = self.ready_queue.popleft()
                core_id = self.select_best_core_for_process(proc)
                
                if core_id is None:
                    # No idle cores, put back in queue
                    self.ready_queue.appendleft(proc)
                    break
                
                self.cores[core_id].assign_process(proc, time)
            
            # Execute on all cores for quantum
            exec_time = self.quantum
            completed_in_cycle = []
            
            for core in self.cores:
                completed_proc = core.execute(exec_time, time)
                if completed_proc:
                    completed_in_cycle.append(completed_proc)
                    self.memory_manager.deallocate(completed_proc)
                    self.completed.append(completed_proc)
            
            # Move preempted processes back to ready queue
            for core in self.cores:
                if core.current_process and core.current_process.remaining_time > 0:
                    self.ready_queue.append(core.current_process)
                    core.current_process = None
                    self.context_switches += 1
            
            time += exec_time
            
            # Progress update
            if time % 20 == 0:
                active_cores = sum(1 for c in self.cores if not c.is_idle())
                mem_usage = self.memory_manager.get_memory_usage()
                print(f"   Time {time:3d}: Active Cores={active_cores}/{self.num_cores}, "
                      f"Memory={mem_usage:.1f}%, Ready={len(self.ready_queue)}, "
                      f"Waiting={len(self.waiting_queue)}")
        
        # Build gantt chart from core histories
        for core in self.cores:
            self.gantt_chart.extend(core.execution_history)
        
        print(f"\n✅ Multi-core scheduling complete!")
    
    def get_metrics(self):
        """Calculate comprehensive metrics"""
        if not self.completed:
            return None
        
        avg_waiting = np.mean([p.waiting_time for p in self.completed])
        avg_turnaround = np.mean([p.turnaround_time for p in self.completed])
        avg_response = np.mean([p.response_time for p in self.completed])
        
        total_burst = sum([p.original_burst for p in self.completed])
        total_time = max([p.completion_time for p in self.completed])
        
        # CPU utilization across all cores
        total_idle = sum([c.idle_time for c in self.cores])
        total_possible = total_time * self.num_cores
        cpu_utilization = ((total_possible - total_idle) / total_possible) * 100 if total_possible > 0 else 0
        
        throughput = len(self.completed) / total_time if total_time > 0 else 0
        
        # Core-specific metrics
        core_utilization = []
        for core in self.cores:
            core_busy_time = total_time - core.idle_time
            core_util = (core_busy_time / total_time) * 100 if total_time > 0 else 0
            core_utilization.append(core_util)
        
        return {
            'avg_waiting_time': avg_waiting,
            'avg_turnaround_time': avg_turnaround,
            'avg_response_time': avg_response,
            'cpu_utilization': cpu_utilization,
            'throughput': throughput,
            'context_switches': self.context_switches,
            'num_cores': self.num_cores,
            'core_utilization': core_utilization,
            'total_time': total_time
        }
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "="*80)
        print(f"📊 MULTI-CORE SCHEDULING RESULTS ({self.num_cores} Cores)")
        print("="*80)
        
        print(f"\n{'PID':<6} {'Core':<6} {'Arrival':<10} {'Burst':<8} "
              f"{'Comp':<8} {'TAT':<8} {'WT':<8}")
        print("-"*80)
        
        for p in sorted(self.completed, key=lambda x: x.pid):
            # Find which core executed this process
            core_id = next((g['core'] for g in self.gantt_chart if g['pid'] == p.pid), 'N/A')
            print(f"{p.pid:<6} {core_id:<6} {p.arrival_time:<10} {p.original_burst:<8} "
                  f"{p.completion_time:<8} {p.turnaround_time:<8} {p.waiting_time:<8}")
        
        metrics = self.get_metrics()
        
        print("\n" + "="*80)
        print("📈 PERFORMANCE METRICS")
        print("="*80)
        print(f"Average Waiting Time:     {metrics['avg_waiting_time']:.2f} units")
        print(f"Average Turnaround Time:  {metrics['avg_turnaround_time']:.2f} units")
        print(f"Average Response Time:    {metrics['avg_response_time']:.2f} units")
        print(f"Overall CPU Utilization:  {metrics['cpu_utilization']:.2f}%")
        print(f"Throughput:               {metrics['throughput']:.4f} proc/unit")
        print(f"Context Switches:         {metrics['context_switches']}")
        print(f"Total Execution Time:     {metrics['total_time']} units")
        
        print(f"\n💻 Per-Core Utilization:")
        for i, util in enumerate(metrics['core_utilization']):
            bar = '█' * int(util / 2)
            print(f"   Core {i}: {util:>6.2f}% {bar}")
        
        print("="*80)


# Demo
if __name__ == "__main__":
    print("="*80)
    print("🖥️  Multi-Core CPU Scheduler Demo")
    print("="*80)
    
    # Create test processes
    processes = [
        Process(1, 0, 24, priority=3, process_size=500, process_type=0, memory_usage=256),
        Process(2, 1, 3, priority=8, process_size=100, process_type=3, memory_usage=64),
        Process(3, 2, 8, priority=5, process_size=300, process_type=1, memory_usage=128),
        Process(4, 3, 12, priority=2, process_size=800, process_type=0, memory_usage=512),
        Process(5, 4, 6, priority=6, process_size=200, process_type=2, memory_usage=192),
        Process(6, 5, 15, priority=4, process_size=600, process_type=0, memory_usage=384),
        Process(7, 6, 5, priority=7, process_size=150, process_type=3, memory_usage=96),
        Process(8, 7, 10, priority=5, process_size=400, process_type=1, memory_usage=256),
    ]
    
    # Create multi-core scheduler
    scheduler = MultiCoreScheduler(
        num_cores=4,
        total_memory=2048,
        use_ml=True,
        quantum=4
    )
    
    scheduler.add_processes(processes)
    scheduler.schedule_multicore_rr()
    scheduler.print_results()
    
    print("\n✨ Multi-core demo complete!")