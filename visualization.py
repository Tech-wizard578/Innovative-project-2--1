"""
visualization.py - Visualization Module for SmartSched
Place this file in: D:\SmartSched\visualization.py
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class SchedulerVisualizer:
    """Creates visualizations for SmartSched"""
    
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        sns.set_style("whitegrid")
    
    def plot_gantt_chart(self, gantt_data, title="SmartSched - Gantt Chart"):
        """Create Gantt chart"""
        if not gantt_data:
            print("⚠️  No Gantt chart data")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        pids = sorted(list(set([g['pid'] for g in gantt_data])))
        pid_to_y = {pid: i for i, pid in enumerate(pids)}
        
        for entry in gantt_data:
            pid = entry['pid']
            start = entry['start']
            duration = entry['end'] - start
            y_pos = pid_to_y[pid]
            
            color = self.colors[pid % len(self.colors)]
            ax.barh(y_pos, duration, left=start, height=0.6,
                   color=color, edgecolor='black', linewidth=1.5, alpha=0.8)
            
            if duration >= 2:
                ax.text(start + duration/2, y_pos, f'P{pid}',
                       ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Time Units', fontsize=12, fontweight='bold')
        ax.set_ylabel('Process ID', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels([f'P{pid}' for pid in pids])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, results_dict):
        """Compare algorithms"""
        if not results_dict:
            print("⚠️  No comparison data")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🏆 SmartSched vs Traditional Algorithms', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        algorithms = list(results_dict.keys())
        metrics = ['avg_waiting_time', 'avg_turnaround_time', 
                   'cpu_utilization', 'throughput']
        titles = ['Average Waiting Time ⏱️', 'Average Turnaround Time 🔄',
                 'CPU Utilization 💻', 'Throughput 📊']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            values = [results_dict[alg][metric] for alg in algorithms]
            
            bars = ax.bar(algorithms, values, color=self.colors[:len(algorithms)],
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            for i, alg in enumerate(algorithms):
                if 'SMART' in alg.upper():
                    bars[i].set_color('#FFD700')
                    bars[i].set_edgecolor('red')
                    bars[i].set_linewidth(3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_ml_prediction_accuracy(self, actual, predicted):
        """Visualize ML prediction accuracy"""
        if not actual or not predicted:
            print("⚠️  Invalid prediction data")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(actual, predicted, alpha=0.6, s=100, 
                       c=self.colors[0], edgecolor='black', linewidth=1.5)
        
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
        
        mae = np.mean(np.abs(errors))
        accuracy = (1 - mae/np.mean(actual)) * 100
        
        fig.text(0.5, 0.02, f'MAE: {mae:.2f} | Accuracy: {accuracy:.1f}%',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        return fig