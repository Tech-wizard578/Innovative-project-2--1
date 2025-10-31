"""
rl_scheduler.py - RL Meta-Scheduler (AI-Powered Algorithm Selection)
Dynamically chooses the best scheduling algorithm based on workload
"""

import numpy as np
import pickle
import os
from smart_scheduler import SmartScheduler, Process


class RLMetaScheduler:
    """
    Reinforcement Learning Meta-Scheduler
    Analyzes workload and intelligently selects the best scheduling algorithm
    """
    
    def __init__(self):
        self.rl_agent = None
        self.available_algorithms = ['FCFS', 'SJF', 'SRTF', 'PRIORITY', 'RR', 'SMART_HYBRID']
        self.load_rl_agent()
    
    def load_rl_agent(self):
        """Load pre-trained RL agent"""
        agent_path = 'models/rl_agent.pkl'
        
        if os.path.exists(agent_path):
            try:
                with open(agent_path, 'rb') as f:
                    data = pickle.load(f)
                    self.rl_agent = data
                    print("✅ Loaded RL agent for intelligent algorithm selection")
                    return True
            except Exception as e:
                print(f"⚠️  Could not load RL agent: {e}")
                return False
        else:
            print("⚠️  RL agent not found. Train it first with train_models.py")
            return False
    
    def analyze_workload(self, processes):
        """
        Analyze workload characteristics to create state representation
        
        Returns:
            state: tuple representing workload characteristics
        """
        if not processes:
            return (0, 0, 0, 0)
        
        # Calculate workload features
        burst_times = [p.burst_time for p in processes]
        priorities = [p.priority for p in processes]
        arrival_times = [p.arrival_time for p in processes]
        process_types = [p.process_type for p in processes]
        
        # State features
        queue_length = len(processes)
        avg_burst = np.mean(burst_times)
        burst_variance = np.var(burst_times)
        priority_range = max(priorities) - min(priorities) if len(priorities) > 1 else 0
        avg_priority = np.mean(priorities)
        arrival_spread = max(arrival_times) - min(arrival_times) if len(arrival_times) > 1 else 0
        
        # Determine workload characteristics
        interactive_ratio = sum(1 for pt in process_types if pt == 3) / len(processes)
        cpu_bound_ratio = sum(1 for pt in process_types if pt == 0) / len(processes)
        
        # Create state (discretized for Q-learning)
        state = (
            min(20, queue_length),                    # Queue length (0-20)
            int(min(50, avg_burst)),                  # Avg burst (0-50)
            int(min(10, priority_range)),             # Priority range (0-10)
            int(min(100, arrival_spread))             # Arrival spread (0-100)
        )
        
        # Store detailed analysis
        self.workload_analysis = {
            'queue_length': queue_length,
            'avg_burst': avg_burst,
            'burst_variance': burst_variance,
            'priority_range': priority_range,
            'avg_priority': avg_priority,
            'interactive_ratio': interactive_ratio,
            'cpu_bound_ratio': cpu_bound_ratio,
            'arrival_spread': arrival_spread
        }
        
        return state
    
    def select_algorithm(self, processes):
        """
        Use RL agent to select best algorithm for given workload
        
        Args:
            processes: List of Process objects
        
        Returns:
            selected_algorithm: Name of chosen algorithm
            confidence: Confidence score (0-1)
            reasoning: Explanation of choice
        """
        # Analyze workload
        state = self.analyze_workload(processes)
        
        print("\n🧠 RL META-SCHEDULER ANALYSIS")
        print("="*60)
        print(f"Workload State: {state}")
        print(f"\nDetailed Analysis:")
        for key, value in self.workload_analysis.items():
            print(f"   {key:<20}: {value:.2f}")
        
        # Use RL agent if available
        if self.rl_agent and 'q_table' in self.rl_agent:
            q_table = self.rl_agent['q_table']
            
            # Get Q-values for this state
            if state in q_table:
                q_values = q_table[state]
                best_action_idx = np.argmax(q_values)
                confidence = q_values[best_action_idx] / (np.sum(np.abs(q_values)) + 1e-10)
                
                selected_algorithm = self.available_algorithms[best_action_idx % len(self.available_algorithms)]
                
                print(f"\n🎯 RL Agent Decision:")
                print(f"   Q-values: {q_values}")
                print(f"   Selected: {selected_algorithm}")
                print(f"   Confidence: {confidence:.2%}")
            else:
                # State not seen during training - use heuristic
                selected_algorithm, confidence = self._heuristic_selection()
                print(f"\n⚠️  State not in Q-table. Using heuristic.")
                print(f"   Selected: {selected_algorithm}")
        else:
            # Fallback to intelligent heuristic
            selected_algorithm, confidence = self._heuristic_selection()
            print(f"\n💡 Using Intelligent Heuristic:")
            print(f"   Selected: {selected_algorithm}")
            print(f"   Confidence: {confidence:.2%}")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(selected_algorithm)
        
        print(f"\n📝 Reasoning: {reasoning}")
        print("="*60)
        
        return selected_algorithm, confidence, reasoning
    
    def _heuristic_selection(self):
        """
        Intelligent heuristic for algorithm selection when RL agent unavailable
        Based on workload characteristics
        """
        analysis = self.workload_analysis
        
        # Decision rules based on OS scheduling theory
        if analysis['interactive_ratio'] > 0.6:
            # Mostly interactive processes - need quick response
            return 'RR', 0.85
        
        elif analysis['cpu_bound_ratio'] > 0.7 and analysis['burst_variance'] > 100:
            # CPU-bound with high variance - SRTF minimizes waiting time
            return 'SRTF', 0.90
        
        elif analysis['priority_range'] > 6:
            # High priority variation - use priority scheduling
            return 'PRIORITY', 0.88
        
        elif analysis['burst_variance'] < 50 and analysis['avg_burst'] < 15:
            # Low variance, short jobs - SJF is optimal
            return 'SJF', 0.92
        
        elif analysis['queue_length'] < 5:
            # Small workload - FCFS is simple and effective
            return 'FCFS', 0.75
        
        else:
            # Complex workload - use our smart hybrid
            return 'SMART_HYBRID', 0.95
    
    def _generate_reasoning(self, algorithm):
        """Generate human-readable reasoning for algorithm choice"""
        analysis = self.workload_analysis
        
        reasoning_map = {
            'FCFS': f"Simple workload with {analysis['queue_length']} processes. FCFS provides predictable, fair scheduling.",
            
            'SJF': f"Low burst variance ({analysis['burst_variance']:.1f}) and short average burst time ({analysis['avg_burst']:.1f}). SJF minimizes average waiting time.",
            
            'SRTF': f"High CPU-bound ratio ({analysis['cpu_bound_ratio']:.1%}) with variable burst times. SRTF optimizes for shortest remaining time.",
            
            'PRIORITY': f"High priority range ({analysis['priority_range']}) detected. Priority scheduling ensures important processes run first.",
            
            'RR': f"High interactive process ratio ({analysis['interactive_ratio']:.1%}). Round-robin provides good response time and fairness.",
            
            'SMART_HYBRID': f"Complex workload detected. ML-powered hybrid scheduler adapts dynamically for optimal performance across all metrics."
        }
        
        return reasoning_map.get(algorithm, "Optimal algorithm selected based on workload analysis.")
    
    def run_with_selected_algorithm(self, processes, use_ml=True):
        """
        Run scheduling with RL-selected algorithm
        
        Args:
            processes: List of Process objects
            use_ml: Whether to use ML predictions in SMART_HYBRID
        
        Returns:
            results: Dict with metrics and selected algorithm
        """
        # Select algorithm
        selected_algorithm, confidence, reasoning = self.select_algorithm(processes)
        
        # Create scheduler
        scheduler = SmartScheduler(
            use_ml=(use_ml and selected_algorithm == 'SMART_HYBRID'),
            quantum=4,
            algorithm=selected_algorithm
        )
        
        # Add processes
        scheduler.add_processes_batch(processes)
        
        # Run scheduling
        print(f"\n▶️  Executing {selected_algorithm} scheduling...")
        metrics = scheduler.run(selected_algorithm)
        
        # Return comprehensive results
        return {
            'algorithm': selected_algorithm,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': metrics,
            'scheduler': scheduler,
            'workload_analysis': self.workload_analysis
        }
    
    def compare_with_rl_selection(self, processes):
        """
        Compare RL-selected algorithm vs all algorithms
        Shows the intelligence of RL selection
        """
        print("\n" + "="*70)
        print("🤖 RL META-SCHEDULER VS ALL ALGORITHMS")
        print("="*70)
        
        # Get RL selection
        rl_results = self.run_with_selected_algorithm(processes)
        
        print(f"\n✅ RL Selected: {rl_results['algorithm']}")
        print(f"   Confidence: {rl_results['confidence']:.1%}")
        print(f"   Reasoning: {rl_results['reasoning']}")
        
        # Compare with all algorithms
        all_results = {'RL_DISPATCHER': rl_results['metrics']}
        
        for algo in ['FCFS', 'SJF', 'RR', 'SMART_HYBRID']:
            if algo != rl_results['algorithm']:
                print(f"\n🔄 Testing {algo}...")
                
                # Fresh copies
                proc_copies = [
                    Process(p.pid, p.arrival_time, p.burst_time, p.priority,
                           p.process_size, p.process_type, p.memory_usage, p.cpu_affinity)
                    for p in processes
                ]
                
                scheduler = SmartScheduler(use_ml=(algo=='SMART_HYBRID'), quantum=4)
                scheduler.add_processes_batch(proc_copies)
                metrics = scheduler.run(algo)
                all_results[algo] = metrics
        
        # Print comparison
        print("\n" + "="*70)
        print("📊 PERFORMANCE COMPARISON")
        print("="*70)
        print(f"{'Algorithm':<20} {'Avg WT':<12} {'Avg TAT':<12} {'CPU %':<10}")
        print("-"*70)
        
        for algo, metrics in all_results.items():
            marker = " ⭐" if algo == 'RL_DISPATCHER' else ""
            print(f"{algo:<20} {metrics['avg_waiting_time']:>10.2f}  "
                  f"{metrics['avg_turnaround_time']:>10.2f}  "
                  f"{metrics['cpu_utilization']:>8.2f}{marker}")
        
        print("="*70)
        
        # Validate RL choice
        best_wt = min(all_results.items(), key=lambda x: x[1]['avg_waiting_time'])
        
        if best_wt[0] == 'RL_DISPATCHER':
            print(f"\n🏆 RL DISPATCHER MADE THE OPTIMAL CHOICE!")
        else:
            wt_diff = (all_results['RL_DISPATCHER']['avg_waiting_time'] - 
                      best_wt[1]['avg_waiting_time'])
            print(f"\n📊 RL Dispatcher within {wt_diff:.2f} units of optimal")
            print(f"   (Best was {best_wt[0]} with {best_wt[1]['avg_waiting_time']:.2f})")
        
        return all_results, rl_results


# Demo
if __name__ == "__main__":
    print("="*70)
    print("🤖 RL Meta-Scheduler Demo")
    print("="*70)
    
    # Create test workload
    processes = [
        Process(1, 0, 24, priority=3, process_size=500, process_type=0),
        Process(2, 1, 3, priority=8, process_size=100, process_type=3),
        Process(3, 2, 8, priority=5, process_size=300, process_type=1),
        Process(4, 3, 12, priority=2, process_size=800, process_type=0),
        Process(5, 4, 6, priority=6, process_size=200, process_type=2),
    ]
    
    # Create meta-scheduler
    meta_scheduler = RLMetaScheduler()
    
    # Run with RL selection and compare
    all_results, rl_results = meta_scheduler.compare_with_rl_selection(processes)
    
    print("\n✨ RL Meta-Scheduler Demo Complete!")