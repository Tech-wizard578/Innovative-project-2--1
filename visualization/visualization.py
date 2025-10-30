"""
visualization.py - Complete Visualization Module for SmartSched
Creates beautiful charts and graphs for process scheduling analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

class SchedulerVisualizer:
    """
    Creates beautiful visualizations for SmartSched
    """
    
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_gantt_chart(self, gantt_data, title="SmartSched - Gantt Chart"):
        """
        Create an interactive Gantt chart
        
        Args:
            gantt_data: list of dicts with 'pid', 'start', 'end'
            title: Chart title
        """
        if not gantt_data:
            print("⚠️  No Gantt chart data available")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get unique PIDs
        pids = sorted(list(set([g['pid'] for g in gantt_data])))
        pid_to_y = {pid: i for i, pid in enumerate(pids)}
        
        # Plot each process execution
        for entry in gantt_data:
            pid = entry['pid']
            start = entry['start']
            duration = entry['end'] - start
            y_pos = pid_to_y[pid]
            
            # Create bar
            color = self.colors[pid % len(self.colors)]
            ax.barh(y_pos, duration, left=start, height=0.6,
                   color=color, edgecolor='black', linewidth=1.5,
                   alpha=0.8)
            
            # Add text label
            if duration >= 2:  # Only show text if bar is wide enough
                ax.text(start + duration/2, y_pos, f'P{pid}',
                       ha='center', va='center', fontweight='bold',
                       fontsize=10, color='black')
        
        # Styling
        ax.set_xlabel('Time Units', fontsize=12, fontweight='bold')
        ax.set_ylabel('Process ID', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels([f'P{pid}' for pid in pids])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, results_dict):
        """
        Compare multiple scheduling algorithms
        
        Args:
            results_dict: {'Algorithm Name': metrics_dict, ...}
        """
        if not results_dict:
            print("⚠️  No comparison data available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🏆 SmartSched vs Traditional Algorithms', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        algorithms = list(results_dict.keys())
        metrics = ['avg_waiting_time', 'avg_turnaround_time', 
                   'cpu_utilization', 'throughput']
        titles = ['Average Waiting Time ⏱️ (Lower is Better)', 
                 'Average Turnaround Time 🔄 (Lower is Better)',
                 'CPU Utilization 💻 (Higher is Better)', 
                 'Throughput 📊 (Higher is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            values = [results_dict[alg][metric] for alg in algorithms]
            
            # Create bar chart
            bars = ax.bar(algorithms, values, color=self.colors[:len(algorithms)],
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Highlight SmartSched
            for i, alg in enumerate(algorithms):
                if 'SMART' in alg.upper() or 'HYBRID' in alg.upper():
                    bars[i].set_color('#FFD700')  # Gold color
                    bars[i].set_edgecolor('red')
                    bars[i].set_linewidth(3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_ml_prediction_accuracy(self, actual, predicted):
        """
        Visualize ML prediction accuracy
        
        Args:
            actual: list of actual burst times
            predicted: list of predicted burst times
        """
        if not actual or not predicted or len(actual) != len(predicted):
            print("⚠️  Invalid prediction data")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(actual, predicted, alpha=0.6, s=100, 
                       c=self.colors[0], edgecolor='black', linewidth=1.5)
        
        # Perfect prediction line
        max_val = max(max(actual), max(predicted))
        axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, 
                    label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Burst Time', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Burst Time', fontsize=12, fontweight='bold')
        axes[0].set_title('🎯 ML Prediction Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Error distribution
        errors = np.array(predicted) - np.array(actual)
        axes[1].hist(errors, bins=15, color=self.colors[2], 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, 
                       label='Zero Error')
        axes[1].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('📊 Error Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Calculate metrics
        mae = np.mean(np.abs(errors))
        accuracy = (1 - mae/np.mean(actual)) * 100
        
        fig.text(0.5, 0.02, f'Mean Absolute Error: {mae:.2f} | Accuracy: {accuracy:.1f}%',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_process_timeline(self, processes):
        """
        Create a detailed process timeline showing arrival, execution, and completion
        """
        if not processes:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, proc in enumerate(processes):
            y_pos = i
            
            # Waiting period (arrival to start)
            if proc.start_time > proc.arrival_time:
                wait_duration = proc.start_time - proc.arrival_time
                ax.barh(y_pos, wait_duration, left=proc.arrival_time, 
                       height=0.5, color='lightgray', alpha=0.5,
                       edgecolor='black', linewidth=1)
            
            # Execution period
            exec_duration = proc.original_burst
            ax.barh(y_pos, exec_duration, left=proc.start_time,
                   height=0.5, color=self.colors[proc.pid % len(self.colors)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add labels
            ax.text(proc.arrival_time, y_pos, f'A', 
                   ha='right', va='center', fontsize=8, fontweight='bold')
            ax.text(proc.completion_time, y_pos, f'C',
                   ha='left', va='center', fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(processes)))
        ax.set_yticklabels([f'P{p.pid}' for p in processes])
        ax.set_xlabel('Time Units', fontsize=12, fontweight='bold')
        ax.set_ylabel('Process', fontsize=12, fontweight='bold')
        ax.set_title('Process Timeline (A=Arrival, C=Completion)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_radar(self, results_dict):
        """
        Create radar chart comparing metrics across algorithms
        """
        if not results_dict or len(results_dict) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Categories
        categories = ['Wait Time\n(Inverted)', 'Turnaround\n(Inverted)', 
                     'CPU Usage', 'Throughput\n(×1000)', 'Response\n(Inverted)']
        N = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot each algorithm
        for i, (algo, metrics) in enumerate(results_dict.items()):
            # Normalize values (higher is better for all)
            values = [
                max(0, 100 - metrics.get('avg_waiting_time', 50)),
                max(0, 100 - metrics.get('avg_turnaround_time', 50)),
                metrics.get('cpu_utilization', 0),
                min(100, metrics.get('throughput', 0) * 1000),
                max(0, 100 - metrics.get('avg_response_time', 50))
            ]
            values += values[:1]
            
            color = self.colors[i % len(self.colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Algorithm Performance Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, scheduler, filename_prefix='smartsched'):
        """Save all visualizations to files"""
        try:
            # Gantt Chart
            fig1 = self.plot_gantt_chart(scheduler.gantt_chart)
            if fig1:
                fig1.savefig(f'{filename_prefix}_gantt.png', dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {filename_prefix}_gantt.png")
            
            # ML Accuracy
            if hasattr(scheduler, 'completed') and scheduler.completed:
                actual = [p.original_burst for p in scheduler.completed]
                predicted = [p.predicted_burst for p in scheduler.completed 
                            if hasattr(p, 'predicted_burst') and p.predicted_burst]
                
                if len(predicted) == len(actual):
                    fig2 = self.plot_ml_prediction_accuracy(actual, predicted)
                    if fig2:
                        fig2.savefig(f'{filename_prefix}_ml_accuracy.png', 
                                   dpi=300, bbox_inches='tight')
                        print(f"✅ Saved: {filename_prefix}_ml_accuracy.png")
            
            print(f"\n💾 All visualizations saved with prefix: {filename_prefix}")
        except Exception as e:
            print(f"⚠️  Error saving plots: {e}")


# Demo usage
if __name__ == "__main__":
    print("🎨 SmartSched Visualization Demo")
    print("="*60)
    
    # Sample Gantt data
    gantt_data = [
        {'pid': 1, 'start': 0, 'end': 4},
        {'pid': 2, 'start': 4, 'end': 7},
        {'pid': 3, 'start': 7, 'end': 10},
        {'pid': 1, 'start': 10, 'end': 14},
        {'pid': 4, 'start': 14, 'end': 18},
        {'pid': 1, 'start': 18, 'end': 24},
    ]
    
    # Sample comparison data
    results_dict = {
        'FCFS': {
            'avg_waiting_time': 15.2,
            'avg_turnaround_time': 28.5,
            'avg_response_time': 12.3,
            'cpu_utilization': 75.3,
            'throughput': 0.0425
        },
        'SJF': {
            'avg_waiting_time': 10.8,
            'avg_turnaround_time': 23.1,
            'avg_response_time': 9.5,
            'cpu_utilization': 82.1,
            'throughput': 0.0461
        },
        'RR': {
            'avg_waiting_time': 12.5,
            'avg_turnaround_time': 25.8,
            'avg_response_time': 4.2,
            'cpu_utilization': 78.9,
            'throughput': 0.0438
        },
        'SmartSched': {
            'avg_waiting_time': 8.3,
            'avg_turnaround_time': 19.7,
            'avg_response_time': 6.1,
            'cpu_utilization': 89.4,
            'throughput': 0.0512
        }
    }
    
    # Sample ML data
    actual_bursts = [24, 3, 3, 12, 6, 8, 15, 5]
    predicted_bursts = [22, 4, 3, 13, 7, 9, 14, 5]
    
    # Create visualizer
    viz = SchedulerVisualizer()
    
    # Generate plots
    print("\n📊 Generating Gantt Chart...")
    fig1 = viz.plot_gantt_chart(gantt_data, "SmartSched - Process Execution")
    
    print("📊 Generating Comparison Chart...")
    fig2 = viz.plot_comparison(results_dict)
    
    print("📊 Generating ML Accuracy Plot...")
    fig3 = viz.plot_ml_prediction_accuracy(actual_bursts, predicted_bursts)
    
    print("📊 Generating Radar Chart...")
    fig4 = viz.plot_metrics_radar(results_dict)
    
    print("\n✨ All visualizations generated!")
    print("💡 Close plot windows to exit...")
    
    plt.show()
    
    print("\n✅ Visualization demo complete!")