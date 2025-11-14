"""
multicore_scheduler.py - Multi-Core CPU Scheduler
Simulates parallel processing on multiple CPU cores with memory management
"""

import numpy as np
from collections import deque
import asyncio
import time

# --- DEFINE PROCESS CLASS HERE ---
# (This removes the need to import from smart_scheduler.py)
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
        
        # Timing metrics
        self.start_time = -1
        self.completion_time = -1
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = -1
        self.predicted_burst = burst_time  # Default to actual burst time

    def get_metrics(self):
        """Returns a dictionary of the process's calculated metrics."""
        return {
            "pid": self.pid,
            "wait_time": max(0, self.waiting_time),
            "response_time": max(0, self.response_time),
            "turnaround_time": max(0, self.turnaround_time)
        }


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
        # Calculate response time ONLY if it hasn't been set
        if process.response_time == -1:
            process.response_time = current_time - process.arrival_time
    
    def execute(self, time_slice, current_time):
        """Execute current process for time_slice"""
        if self.current_process is None:
            self.idle_time += time_slice
            return None
        
        # Use duration for easier Gantt charting
        exec_time = min(time_slice, self.current_process.remaining_time)
        
        self.execution_history.append({
            'pid': self.current_process.pid,
            'core': self.core_id,
            'start': current_time,
            'duration': exec_time 
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
    
    def __init__(self, num_cores=4, total_memory=2048, quantum=4):
        self.num_cores = num_cores
        self.cores = [Core(i) for i in range(num_cores)]
        self.memory_manager = MemoryManager(total_memory)
        self.quantum = quantum
        
        self.processes = []
        self.ready_queue = deque()
        self.waiting_queue = deque()  # Processes waiting for memory
        self.completed_processes = []
        self.gantt_chart = []
        self.context_switches = 0
        self.current_time = 0
        self.log_function = print  # Default to regular print
    
    def add_processes(self, processes):
        """Add processes to scheduler"""
        self.processes = sorted(processes, key=lambda x: x.arrival_time)
    
    def select_best_core_for_process(self, process):
        """
        Select best core for process based on cpu_affinity
        Returns core_id or None if all busy
        """
        idle_cores = [c for c in self.cores if c.is_idle()]
        
        if not idle_cores:
            return None
        
        # Check CPU affinity
        if process.cpu_affinity < len(self.cores):
            preferred_core = self.cores[process.cpu_affinity]
            if preferred_core.is_idle():
                return preferred_core.core_id
        
        # Return first idle core
        return idle_cores[0].core_id
    
    # --- FIX 1: RENAMED FUNCTION TO 'run_simulation' ---
    async def run_simulation(self, processes):
        """
        Run the scheduling simulation asynchronously.
        This is the main function called by main.py
        """
        self.add_processes(processes)
        
        # Reset state for a new simulation
        self.ready_queue = deque()
        self.waiting_queue = deque()
        self.completed_processes = []
        self.gantt_chart = []
        self.context_switches = 0
        self.current_time = 0
        self.cores = [Core(i) for i in range(self.num_cores)]
        self.memory_manager = MemoryManager(self.memory_manager.total_memory)

        # Call the actual async scheduler logic
        return await self.schedule_multicore_rr_async()
    
    async def schedule_multicore_rr_async(self):
        """
        Multi-core Round Robin with Memory Management (Async version)
        """
        remaining = self.processes.copy()
        
        await self.log_function(f"⚙️  Multi-Core Scheduling on {self.num_cores} cores")
        await self.log_function(f"   Total Memory: {self.memory_manager.total_memory}MB")
        await self.log_function(f"   Quantum: {self.quantum} units\n")
        
        while remaining or self.ready_queue or self.waiting_queue or any(not c.is_idle() for c in self.cores):
            # Add newly arrived processes to waiting queue
            while remaining and remaining[0].arrival_time <= self.current_time:
                proc = remaining.pop(0)
                self.waiting_queue.append(proc)
                await self.log_function(f"📥 Process P{proc.pid} arrived at time {self.current_time}")
            
            # Try to allocate memory for waiting processes
            successfully_allocated = []
            for proc in list(self.waiting_queue):
                if self.memory_manager.allocate(proc):
                    self.ready_queue.append(proc)
                    successfully_allocated.append(proc)
                    await self.log_function(f"🧠 Allocated memory for P{proc.pid}")
            
            for proc in successfully_allocated:
                self.waiting_queue.remove(proc)
            
            # Assign processes to idle cores
            while self.ready_queue:
                proc = self.ready_queue.popleft()
                core_id = self.select_best_core_for_process(proc)
                
                if core_id is None:
                    self.ready_queue.appendleft(proc)
                    break
                
                self.cores[core_id].assign_process(proc, self.current_time)
                await self.log_function(f"⚡ Assigned P{proc.pid} to Core {core_id}")
            
            # Execute on all cores for quantum
            exec_time = self.quantum
            completed_in_cycle = []
            
            for core in self.cores:
                completed_proc = core.execute(exec_time, self.current_time)
                if completed_proc:
                    completed_in_cycle.append(completed_proc)
                    self.memory_manager.deallocate(completed_proc)
                    self.completed_processes.append(completed_proc)
                    await self.log_function(f"✅ Completed P{completed_proc.pid} at time {self.current_time + exec_time}")
            
            # Move preempted processes back to ready queue
            for core in self.cores:
                if core.current_process and core.current_process.remaining_time > 0:
                    self.ready_queue.append(core.current_process)
                    await self.log_function(f"🔁 Preempted P{core.current_process.pid}")
                    core.current_process = None
                    self.context_switches += 1
            
            self.current_time += exec_time
            
            # Progress update
            if self.current_time % 20 == 0 and self.current_time > 0:
                active_cores = sum(1 for c in self.cores if not c.is_idle())
                mem_usage = self.memory_manager.get_memory_usage()
                await self.log_function(f"   Time {self.current_time:3d}: Active Cores={active_cores}/{self.num_cores}, "
                                      f"Memory={mem_usage:.1f}%, Ready={len(self.ready_queue)}, "
                                      f"Waiting={len(self.waiting_queue)}")
            
            # Small delay to make logs readable
            await asyncio.sleep(0.01) # Reduced delay for speed
        
        # Build gantt chart from core histories
        self.gantt_chart = []
        for core in self.cores:
            self.gantt_chart.extend(core.execution_history)
        
        await self.log_function(f"\n✅ Multi-core scheduling complete!")

        # --- FIX 2: RETURN THE FULL METRICS OBJECT ---
        metrics = self.calculate_metrics()
        metrics['gantt_chart'] = self.gantt_chart
        metrics['process_metrics'] = [p.get_metrics() for p in self.completed_processes]
        metrics['total_time'] = self.current_time
        return metrics
        # --- END FIX 2 ---
    
    # --- FIX 3: REPLACE THIS ENTIRE FUNCTION ---
    def calculate_metrics(self):
        """Calculate and return simulation metrics as numbers"""
        if not self.completed_processes:
            return {
                "avg_wait_time": 0,
                "avg_response_time": 0,
                "avg_turnaround_time": 0,
                "cpu_utilization": 0,
                "throughput": 0,
                "context_switches": 0,
                "core_utilization": [0] * self.num_cores
            }
        
        wait_times = [max(0, p.waiting_time) for p in self.completed_processes]
        response_times = [max(0, p.response_time) for p in self.completed_processes]
        turnaround_times = [max(0, p.turnaround_time) for p in self.completed_processes]
        
        avg_waiting = np.mean(wait_times) if wait_times else 0
        avg_response = np.mean(response_times) if response_times else 0
        avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0
        
        if self.current_time == 0:
             return {
                "avg_wait_time": 0, "avg_response_time": 0, "avg_turnaround_time": 0,
                "cpu_utilization": 0, "throughput": 0, "context_switches": 0,
                "core_utilization": [0] * self.num_cores
            }

        total_idle_time = sum(c.idle_time for c in self.cores)
        total_system_time = self.current_time * self.num_cores
        
        # Handle division by zero if total_system_time is 0
        cpu_utilization = ((total_system_time - total_idle_time) / total_system_time) * 100 if total_system_time > 0 else 0
        
        core_util = []
        for core in self.cores:
            core_work_time = self.current_time - core.idle_time
            core_util.append((core_work_time / self.current_time) * 100 if self.current_time > 0 else 0)
            
        throughput = len(self.completed_processes) / self.current_time if self.current_time > 0 else 0
        
        return {
            "avg_wait_time": max(0, avg_waiting),
            "avg_response_time": max(0, avg_response),
            "avg_turnaround_time": max(0, avg_turnaround),
            "cpu_utilization": max(0, cpu_utilization),
            "throughput": max(0, throughput),
            "context_switches": max(0, self.context_switches),
            "core_utilization": [max(0, util) for util in core_util]
        }
    # --- END FIX 3 ---

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
    ]
    
    # Create multi-core scheduler
    scheduler = MultiCoreScheduler(
        num_cores=2,
        total_memory=2048,
        quantum=4
    )
    
    # Simple async runner for demo
    async def run_demo():
        # Call the correct function name
        metrics = await scheduler.run_simulation(processes)
        print("Demo completed!")
        print("Metrics:", metrics)
    
    import asyncio
    asyncio.run(run_demo())
    
    print("\n✨ Multi-core demo complete!")