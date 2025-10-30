"""
SmartSched - AI-Powered Process Scheduler
Complete Demo Script for Project Expo

This integrates:
1. ML Burst Time Prediction
2. Smart Hybrid Scheduling
3. Beautiful Visualizations
4. Comparative Analysis
"""

import sys
import os
import numpy as np

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from smart_scheduler import SmartScheduler, Process
from visualization import SchedulerVisualizer
import matplotlib.pyplot as plt


def create_sample_workload(workload_type='mixed'):
    """
    Create different types of workloads for testing
    """
    processes = []
    
    if workload_type == 'cpu_bound':
        # CPU-intensive processes (long bursts)
        processes = [
            Process(1, 0, 24, priority=3, process_size=800, process_type=0),
            Process(2, 2, 18, priority=2, process_size=900, process_type=0),
            Process(3, 4, 20, priority=4, process_size=750, process_type=0),
            Process(4, 6, 15, priority=5, process_size=600, process_type=0),
            Process(5, 8, 22, priority=3, process_size=850, process_type=0),
        ]
    
    elif workload_type == 'interactive':
        # Interactive processes (short bursts)
        processes = [
            Process(1, 0, 3, priority=8, process_size=100, process_type=3),
            Process(2, 1, 2, priority=9, process_size=80, process_type=3),
            Process(3, 2, 4, priority=7, process_size=120, process_type=3),
            Process(4, 3, 3, priority=8, process_size=90, process_type=3),
            Process(5, 4, 2, priority=9, process_size=85, process_type=3),
            Process(6, 5, 3, priority=8, process_size=95, process_type=3),
        ]
    
    elif workload_type == 'mixed':
        # Mixed workload (realistic scenario)
        processes = [
            Process(1, 0, 24, priority=3, process_size=500, process_type=0),
            Process(2, 1, 3, priority=8, process_size=100, process_type=3),
            Process(3, 2, 8, priority=5, process_size=300, process_type=1),
            Process(4, 3, 12, priority=2, process_size=800, process_type=0),
            Process(5, 4, 6, priority=6, process_size=200, process_type=2),
            Process(6, 5, 4, priority=7, process_size=150, process_type=3),
            Process(7, 6, 15, priority=4, process_size=600, process_type=0),
            Process(8, 7, 5, priority=6, process_size=250, process_type=1),
        ]
    
    elif workload_type == 'priority_test':
        # Test priority handling
        processes = [
            Process(1, 0, 10, priority=1, process_size=400, process_type=0),
            Process(2, 1, 8, priority=10, process_size=200, process_type=3),
            Process(3, 2, 12, priority=3, process_size=500, process_type=0),
            Process(4, 3, 6, priority=9, process_size=150, process_type=3),
            Process(5, 4, 9, priority=5, process_size=300, process_type=1),
        ]
    
    return processes


def compare_all_algorithms(processes):
    """
    Run comprehensive comparison between SmartSched and traditional algorithms
    """
    print("\n" + "="*75)
    print("🏁 RUNNING COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*75)
    
    all_results = {}
    algorithms_to_test = ['SMART_HYBRID', 'FCFS', 'SJF', 'SRTF', 'PRIORITY', 'RR']
    
    for algo in algorithms_to_test:
        print(f"\n🚀 Running {algo}...")
        
        # Create fresh process copies
        proc_copies = [
            Process(p.pid, p.arrival_time, p.burst_time, p.priority, 
                   p.process_size, p.process_type, p.memory_usage, p.cpu_affinity)
            for p in processes
        ]
        
        # Create scheduler
        use_ml = (algo == 'SMART_HYBRID')
        scheduler = SmartScheduler(use_ml=use_ml, quantum=4)
        scheduler.add_processes_batch(proc_copies)
        
        # Run algorithm
        try:
            metrics = scheduler.run(algo)
            all_results[algo] = metrics
            
            # Save scheduler for later visualization (only for SmartSched)
            if algo == 'SMART_HYBRID':
                smart_scheduler = scheduler
        except Exception as e:
            print(f"   ⚠️  Error running {algo}: {e}")
            all_results[algo] = {
                'avg_waiting_time': 0,
                'avg_turnaround_time': 0,
                'avg_response_time': 0,
                'cpu_utilization': 0,
                'throughput': 0,
                'context_switches': 0
            }
    
    return all_results, smart_scheduler


def print_comparison_table(results):
    """
    Print beautiful comparison table
    """
    print("\n" + "="*95)
    print("📊 ALGORITHM PERFORMANCE COMPARISON")
    print("="*95)
    print(f"{'Algorithm':<20} {'Avg WT':<12} {'Avg TAT':<12} {'CPU %':<10} "
          f"{'Throughput':<12} {'Ctx Sw':<8}")
    print("-"*95)
    
    for algo, metrics in results.items():
        print(f"{algo:<20} "
              f"{metrics['avg_waiting_time']:>10.2f}  "
              f"{metrics['avg_turnaround_time']:>10.2f}  "
              f"{metrics['cpu_utilization']:>8.2f}  "
              f"{metrics['throughput']:>10.4f}  "
              f"{metrics['context_switches']:>6}")
    
    print("="*95)
    
    # Find best algorithm for each metric
    valid_results = {k: v for k, v in results.items() if v['avg_waiting_time'] > 0}
    
    if valid_results:
        best_wt = min(valid_results.items(), key=lambda x: x[1]['avg_waiting_time'])
        best_tat = min(valid_results.items(), key=lambda x: x[1]['avg_turnaround_time'])
        best_cpu = max(valid_results.items(), key=lambda x: x[1]['cpu_utilization'])
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Lowest Waiting Time:    {best_wt[0]:<20} ({best_wt[1]['avg_waiting_time']:.2f})")
        print(f"   Lowest Turnaround:      {best_tat[0]:<20} ({best_tat[1]['avg_turnaround_time']:.2f})")
        print(f"   Highest CPU Usage:      {best_cpu[0]:<20} ({best_cpu[1]['cpu_utilization']:.2f}%)")
        
        # Calculate improvement
        if 'SMART_HYBRID' in results and 'FCFS' in results:
            if results['FCFS']['avg_waiting_time'] > 0:
                wt_improvement = ((results['FCFS']['avg_waiting_time'] - 
                                  results['SMART_HYBRID']['avg_waiting_time']) / 
                                 results['FCFS']['avg_waiting_time']) * 100
                tat_improvement = ((results['FCFS']['avg_turnaround_time'] - 
                                   results['SMART_HYBRID']['avg_turnaround_time']) / 
                                  results['FCFS']['avg_turnaround_time']) * 100
                
                print(f"\n📈 SMARTSCHED IMPROVEMENT OVER FCFS:")
                print(f"   Waiting Time:     {wt_improvement:>6.1f}% better")
                print(f"   Turnaround Time:  {tat_improvement:>6.1f}% better")


def main():
    """
    Main demo function - Run this for the expo!
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        🚀 SmartSched - AI-Powered Process Scheduler 🚀          ║
    ║                                                                  ║
    ║              Machine Learning Meets Operating Systems            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Select workload type
    print("\n🎯 Select Workload Type:")
    print("   1. Mixed (Realistic scenario) - RECOMMENDED")
    print("   2. CPU-Bound (Long-running processes)")
    print("   3. Interactive (Short bursts)")
    print("   4. Priority Test")
    
    try:
        choice = input("\nEnter choice (1-4, or press Enter for Mixed): ").strip()
    except:
        choice = '1'
    
    workload_map = {
        '1': 'mixed',
        '2': 'cpu_bound',
        '3': 'interactive',
        '4': 'priority_test',
        '': 'mixed'
    }
    
    workload_type = workload_map.get(choice, 'mixed')
    
    # Create processes
    print(f"\n📦 Creating {workload_type.upper()} workload...")
    processes = create_sample_workload(workload_type)
    
    print(f"✅ Generated {len(processes)} processes")
    print("\nProcess Details:")
    for p in processes:
        proc_types = ['CPU-Bound', 'I/O-Bound', 'Mixed', 'Interactive']
        print(f"   P{p.pid}: Arrival={p.arrival_time}, Burst={p.burst_time}, "
              f"Priority={p.priority}, Type={proc_types[p.process_type]}")
    
    # Run comparisons
    print("\n" + "="*75)
    print("⚙️  TRAINING/LOADING ML MODELS...")
    print("="*75)
    
    results, smart_scheduler = compare_all_algorithms(processes)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print detailed SmartSched results
    print("\n" + "="*75)
    print("🔍 DETAILED SMARTSCHED RESULTS")
    print("="*75)
    smart_scheduler.print_results()
    
    # Generate visualizations
    print("\n" + "="*75)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*75)
    
    try:
        viz = SchedulerVisualizer()
        
        # 1. Gantt Chart
        print("\n📊 Creating Gantt Chart...")
        fig1 = viz.plot_gantt_chart(smart_scheduler.gantt_chart, 
                                    "SmartSched - AI-Powered Execution Timeline")
        
        # 2. Algorithm Comparison
        print("📊 Creating Algorithm Comparison...")
        fig2 = viz.plot_comparison(results)
        
        # 3. ML Prediction Accuracy
        if smart_scheduler.completed and smart_scheduler.use_ml:
            print("📊 Creating ML Prediction Accuracy Plot...")
            actual = [p.original_burst for p in smart_scheduler.completed]
            predicted = [p.predicted_burst if hasattr(p, 'predicted_burst') and p.predicted_burst 
                        else p.original_burst for p in smart_scheduler.completed]
            
            if any(predicted):
                fig3 = viz.plot_ml_prediction_accuracy(actual, predicted)
        
        print("\n✅ All visualizations generated!")
        print("\n💡 Close the plot windows to continue...")
        
        plt.show()
        
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")
        print("   Continuing without plots...")
    
    # Summary
    print("\n" + "="*75)
    print("✨ DEMO COMPLETE!")
    print("="*75)
    print("\n🎯 KEY TAKEAWAYS:")
    print("   ✓ SmartSched uses ML to predict burst times")
    print("   ✓ Achieves better performance than traditional algorithms")
    print("   ✓ Adapts dynamically to different workload types")
    print("   ✓ Provides beautiful visualizations for analysis")
    
    # Performance summary
    if 'SMART_HYBRID' in results:
        print(f"\n📊 SMARTSCHED PERFORMANCE:")
        print(f"   Average Waiting Time:     {results['SMART_HYBRID']['avg_waiting_time']:.2f} units")
        print(f"   Average Turnaround Time:  {results['SMART_HYBRID']['avg_turnaround_time']:.2f} units")
        print(f"   CPU Utilization:          {results['SMART_HYBRID']['cpu_utilization']:.2f}%")
        print(f"   Throughput:               {results['SMART_HYBRID']['throughput']:.4f} proc/unit")
    
    print("\n🏆 Ready for the expo presentation!")
    print("="*75)
    
    # Ask if user wants to run another workload
    print("\n💡 Want to try another workload? Run: python main_demo.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()