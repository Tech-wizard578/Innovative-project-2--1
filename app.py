"""
app.py - Flask Web Application for SmartSched
Provides web interface for process scheduling simulation
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import json

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_scheduler import SmartScheduler, Process
from visualization import SchedulerVisualizer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Global variables
app.config['SECRET_KEY'] = 'smartsched-ai-powered-scheduler'


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/schedule', methods=['POST'])
def schedule_processes():
    """
    API endpoint to schedule processes
    
    Expected JSON:
    {
        "processes": [
            {"pid": 1, "arrival": 0, "burst": 10, "priority": 5, ...},
            ...
        ],
        "algorithm": "SMART_HYBRID",
        "use_ml": true,
        "quantum": 4
    }
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        processes_data = data.get('processes', [])
        algorithm = data.get('algorithm', 'SMART_HYBRID')
        use_ml = data.get('use_ml', True)
        quantum = data.get('quantum', 4)
        
        # Create processes
        processes = []
        for proc_data in processes_data:
            process = Process(
                pid=proc_data['pid'],
                arrival_time=proc_data['arrival'],
                burst_time=proc_data['burst'],
                priority=proc_data.get('priority', 5),
                process_size=proc_data.get('process_size', 100),
                process_type=proc_data.get('process_type', 0),
                memory_usage=proc_data.get('memory_usage', 64),
                cpu_affinity=proc_data.get('cpu_affinity', 0)
            )
            processes.append(process)
        
        # Create and run scheduler
        scheduler = SmartScheduler(use_ml=use_ml, quantum=quantum)
        scheduler.add_processes_batch(processes)
        metrics = scheduler.run(algorithm)
        
        # Prepare response
        response = {
            'success': True,
            'algorithm': algorithm,
            'metrics': metrics,
            'gantt_chart': scheduler.gantt_chart,
            'processes': []
        }
        
        # Add process details
        for proc in scheduler.completed:
            response['processes'].append({
                'pid': proc.pid,
                'arrival_time': proc.arrival_time,
                'burst_time': proc.original_burst,
                'predicted_burst': proc.predicted_burst,
                'completion_time': proc.completion_time,
                'waiting_time': proc.waiting_time,
                'turnaround_time': proc.turnaround_time,
                'response_time': proc.response_time
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """
    Compare multiple scheduling algorithms
    
    Expected JSON:
    {
        "processes": [...],
        "algorithms": ["FCFS", "SJF", "SMART_HYBRID"]
    }
    """
    try:
        data = request.get_json()
        processes_data = data.get('processes', [])
        algorithms = data.get('algorithms', ['FCFS', 'SJF', 'RR', 'SMART_HYBRID'])
        
        results = {}
        
        for algo in algorithms:
            # Create fresh process copies
            processes = []
            for proc_data in processes_data:
                process = Process(
                    pid=proc_data['pid'],
                    arrival_time=proc_data['arrival'],
                    burst_time=proc_data['burst'],
                    priority=proc_data.get('priority', 5),
                    process_size=proc_data.get('process_size', 100),
                    process_type=proc_data.get('process_type', 0)
                )
                processes.append(process)
            
            # Run algorithm
            use_ml = (algo == 'SMART_HYBRID')
            scheduler = SmartScheduler(use_ml=use_ml, quantum=4)
            scheduler.add_processes_batch(processes)
            metrics = scheduler.run(algo)
            
            results[algo] = metrics
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """
    Generate visualization images
    Returns base64 encoded images
    """
    try:
        data = request.get_json()
        viz_type = data.get('type', 'gantt')
        
        if viz_type == 'gantt':
            gantt_data = data.get('gantt_data', [])
            viz = SchedulerVisualizer()
            fig = viz.plot_gantt_chart(gantt_data)
            
        elif viz_type == 'comparison':
            results = data.get('results', {})
            viz = SchedulerVisualizer()
            fig = viz.plot_comparison(results)
            
        elif viz_type == 'ml_accuracy':
            actual = data.get('actual', [])
            predicted = data.get('predicted', [])
            viz = SchedulerVisualizer()
            fig = viz.plot_ml_prediction_accuracy(actual, predicted)
        
        else:
            return jsonify({'success': False, 'error': 'Invalid visualization type'}), 400
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/presets')
def get_presets():
    """Get preset workload configurations"""
    presets = {
        'mixed': [
            {'pid': 1, 'arrival': 0, 'burst': 24, 'priority': 3, 'process_type': 0},
            {'pid': 2, 'arrival': 1, 'burst': 3, 'priority': 8, 'process_type': 3},
            {'pid': 3, 'arrival': 2, 'burst': 8, 'priority': 5, 'process_type': 1},
            {'pid': 4, 'arrival': 3, 'burst': 12, 'priority': 2, 'process_type': 0},
            {'pid': 5, 'arrival': 4, 'burst': 6, 'priority': 6, 'process_type': 2},
        ],
        'cpu_bound': [
            {'pid': 1, 'arrival': 0, 'burst': 24, 'priority': 3, 'process_type': 0},
            {'pid': 2, 'arrival': 2, 'burst': 18, 'priority': 2, 'process_type': 0},
            {'pid': 3, 'arrival': 4, 'burst': 20, 'priority': 4, 'process_type': 0},
            {'pid': 4, 'arrival': 6, 'burst': 15, 'priority': 5, 'process_type': 0},
        ],
        'interactive': [
            {'pid': 1, 'arrival': 0, 'burst': 3, 'priority': 8, 'process_type': 3},
            {'pid': 2, 'arrival': 1, 'burst': 2, 'priority': 9, 'process_type': 3},
            {'pid': 3, 'arrival': 2, 'burst': 4, 'priority': 7, 'process_type': 3},
            {'pid': 4, 'arrival': 3, 'burst': 3, 'priority': 8, 'process_type': 3},
        ]
    }
    
    return jsonify(presets)


# Simple HTML template (inline for simplicity)
@app.route('/simple')
def simple_ui():
    """Simple HTML interface"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartSched - AI Process Scheduler</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #333;
            }
            select, input, button {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            #results {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .metric {
                margin: 10px 0;
                padding: 15px;
                background: white;
                border-left: 4px solid #667eea;
                border-radius: 5px;
            }
            .metric-label {
                font-weight: bold;
                color: #667eea;
            }
            .metric-value {
                float: right;
                color: #333;
                font-size: 1.2em;
            }
            .loading {
                text-align: center;
                color: #667eea;
                font-size: 1.2em;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 SmartSched</h1>
            <p class="subtitle">AI-Powered Process Scheduler</p>
            
            <div class="form-group">
                <label>Select Workload Preset:</label>
                <select id="preset">
                    <option value="mixed">Mixed Workload (Realistic)</option>
                    <option value="cpu_bound">CPU-Bound Processes</option>
                    <option value="interactive">Interactive Processes</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Select Algorithm:</label>
                <select id="algorithm">
                    <option value="SMART_HYBRID">SmartSched (AI-Powered)</option>
                    <option value="FCFS">First Come First Serve</option>
                    <option value="SJF">Shortest Job First</option>
                    <option value="SRTF">Shortest Remaining Time First</option>
                    <option value="PRIORITY">Priority Scheduling</option>
                    <option value="RR">Round Robin</option>
                </select>
            </div>
            
            <button onclick="runScheduler()">▶️ Run Scheduler</button>
            
            <div class="loading" id="loading">⏳ Processing...</div>
            
            <div id="results">
                <h2>📊 Results</h2>
                <div id="metrics"></div>
            </div>
        </div>
        
        <script>
            async function runScheduler() {
                const preset = document.getElementById('preset').value;
                const algorithm = document.getElementById('algorithm').value;
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                // Get preset processes
                const presetsResp = await fetch('/api/presets');
                const presets = await presetsResp.json();
                const processes = presets[preset];
                
                // Run scheduler
                const response = await fetch('/api/schedule', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        processes: processes,
                        algorithm: algorithm,
                        use_ml: algorithm === 'SMART_HYBRID',
                        quantum: 4
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    const metrics = data.metrics;
                    const metricsHTML = `
                        <div class="metric">
                            <span class="metric-label">Average Waiting Time:</span>
                            <span class="metric-value">${metrics.avg_waiting_time.toFixed(2)} units</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Average Turnaround Time:</span>
                            <span class="metric-value">${metrics.avg_turnaround_time.toFixed(2)} units</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">CPU Utilization:</span>
                            <span class="metric-value">${metrics.cpu_utilization.toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Throughput:</span>
                            <span class="metric-value">${metrics.throughput.toFixed(4)} proc/unit</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Context Switches:</span>
                            <span class="metric-value">${metrics.context_switches}</span>
                        </div>
                    `;
                    
                    document.getElementById('metrics').innerHTML = metricsHTML;
                    document.getElementById('results').style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            }
        </script>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        🌐 SmartSched Web Application Starting...  🌐            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    print("\n🚀 Starting Flask server...")
    print("📱 Access the web interface at: http://localhost:5000/simple")
    print("🔌 API endpoints:")
    print("   POST /api/schedule  - Schedule processes")
    print("   POST /api/compare   - Compare algorithms")
    print("   POST /api/visualize - Generate charts")
    print("   GET  /api/presets   - Get workload presets")
    print("\n⚡ Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)