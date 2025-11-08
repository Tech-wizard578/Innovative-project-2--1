"""
app.py - Flask Web Application for SmartSched (Unified UI)
Serves the React/Tailwind SPA and provides a couple of small helper endpoints.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import io
import json
import datetime

# your existing imports for scheduler (keep as-is)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# If these modules exist in your project they will be used.
# If they are missing, the endpoints will still run (you'll need actual implementation).
try:
    from smart_scheduler import SmartScheduler, Process
    from visualization import SchedulerVisualizer
except Exception:
    # Graceful fallback for development - real implementations are expected
    SmartScheduler = None
    Process = None
    SchedulerVisualizer = None

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.config['SECRET_KEY'] = 'smartsched-ai-powered-scheduler-v2'

# Serve the single SPA
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    # All frontend routes served by one template (SPA)
    return render_template('app.html')

# Keep your API endpoints - they assume proper scheduler implementations exist
@app.route('/api/presets', methods=['GET'])
def get_presets():
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


@app.route('/api/schedule', methods=['POST'])
def schedule_processes():
    try:
        data = request.get_json(force=True)
        processes_data = data.get('processes', [])
        algorithm = data.get('algorithm', 'SMART_HYBRID')
        use_ml = data.get('use_ml', True)
        quantum = data.get('quantum', 4)

        # If real SmartScheduler exists, use it
        if SmartScheduler and Process:
            processes = []
            for proc_data in processes_data:
                p = Process(
                    pid=proc_data['pid'],
                    arrival_time=proc_data['arrival'],
                    burst_time=proc_data['burst'],
                    priority=proc_data.get('priority', 5),
                    process_size=proc_data.get('process_size', 100),
                    process_type=proc_data.get('process_type', 0),
                    memory_usage=proc_data.get('memory_usage', 64),
                    cpu_affinity=proc_data.get('cpu_affinity', 0)
                )
                processes.append(p)

            sched = SmartScheduler(use_ml=use_ml, quantum=quantum)
            sched.add_processes_batch(processes)
            metrics = sched.run(algorithm)
            # assume scheduler sets gantt_chart and completed list
            gantt = getattr(sched, 'gantt_chart', [])
            completed = getattr(sched, 'completed', [])
            proc_list = []
            for proc in completed:
                proc_list.append({
                    'pid': getattr(proc, 'pid', None),
                    'arrival_time': getattr(proc, 'arrival_time', None),
                    'burst_time': getattr(proc, 'original_burst', getattr(proc, 'burst_time', None)),
                    'predicted_burst': getattr(proc, 'predicted_burst', None),
                    'completion_time': getattr(proc, 'completion_time', None),
                    'waiting_time': getattr(proc, 'waiting_time', None),
                    'turnaround_time': getattr(proc, 'turnaround_time', None),
                    'response_time': getattr(proc, 'response_time', None)
                })

            return jsonify({'success': True, 'algorithm': algorithm, 'metrics': metrics, 'gantt_chart': gantt, 'processes': proc_list})

        # Fallback: return a mocked response for development/testing
        else:
            # simple simulation metrics
            n = len(processes_data)
            metrics = {
                'avg_waiting_time': 10.2,
                'avg_turnaround_time': 23.5,
                'cpu_utilization': 89.4,
                'throughput': max(0.1, n / 100.0),
                'context_switches': int(n * 3)
            }
            # simple gantt placeholder
            gantt = []
            for p in processes_data:
                gantt.append({'pid': p['pid'], 'start': p.get('arrival', 0), 'duration': p.get('burst', 1)})
            return jsonify({'success': True, 'algorithm': algorithm, 'metrics': metrics, 'gantt_chart': gantt, 'processes': processes_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    try:
        # --- Safely parse request JSON ---
        try:
            data = request.get_json(force=True)
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        processes_data = data.get('processes', [])
        algorithms = data.get('algorithms', ['FCFS', 'SJF', 'RR', 'SMART_HYBRID', 'RL_DISPATCHER'])

        results = {}

        # --- If SmartScheduler backend is available, use it ---
        if 'SmartScheduler' in globals() and 'Process' in globals() and SmartScheduler and Process:
            for algo in algorithms:
                processes = []
                for proc_data in processes_data:
                    process = Process(
                        pid=proc_data.get('pid', 0),
                        arrival_time=proc_data.get('arrival', 0),
                        burst_time=proc_data.get('burst', 1),
                        priority=proc_data.get('priority', 5),
                        process_size=proc_data.get('process_size', 100),
                        process_type=proc_data.get('process_type', 0)
                    )
                    processes.append(process)

                use_ml = (algo in ['SMART_HYBRID', 'RL_DISPATCHER'])
                scheduler = SmartScheduler(use_ml=use_ml, quantum=4)
                scheduler.add_processes_batch(processes)
                metrics = scheduler.run(algo)

                # Ensure consistent metric keys for frontend
                formatted_metrics = {
                    'avg_waiting_time': metrics.get('avg_waiting_time', metrics.get('avgWT', 0)),
                    'avg_turnaround_time': metrics.get('avg_turnaround_time', metrics.get('avgTAT', 0)),
                    'cpu_utilization': metrics.get('cpu_utilization', metrics.get('cpuUtil', 0)),
                    'throughput': metrics.get('throughput', 0),
                    'context_switches': metrics.get('context_switches', 0)
                }

                results[algo] = formatted_metrics

        # --- Otherwise, use mock data (for demo/testing) ---
        else:
            results = {
                'FCFS': {
                    'avg_waiting_time': 106.3,
                    'avg_turnaround_time': 116.8,
                    'cpu_utilization': 82.7,
                    'throughput': 9.41,
                    'context_switches': 0
                },
                'SJF': {
                    'avg_waiting_time': 71.2,
                    'avg_turnaround_time': 81.7,
                    'cpu_utilization': 86.4,
                    'throughput': 14.05,
                    'context_switches': 0
                },
                'RR': {
                    'avg_waiting_time': 95.7,
                    'avg_turnaround_time': 106.2,
                    'cpu_utilization': 83.5,
                    'throughput': 10.45,
                    'context_switches': 3847
                },
                'SMART_HYBRID': {
                    'avg_waiting_time': 57.9,
                    'avg_turnaround_time': 68.4,
                    'cpu_utilization': 91.8,
                    'throughput': 17.27,
                    'context_switches': 892
                },
                'RL_DISPATCHER': {
                    'avg_waiting_time': 52.1,
                    'avg_turnaround_time': 62.7,
                    'cpu_utilization': 94.2,
                    'throughput': 18.92,
                    'context_switches': 765
                }
            }

        # --- Return unified JSON response ---
        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        print("❌ Error in /api/compare:", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400



@app.route('/api/explain', methods=['POST'])
def explain_decision():
    """Simple explain endpoint - returns plain-text explanation of the AI decision (extend to real explainers)"""
    try:
        data = request.get_json(force=True)
        # If you have a model producing explanation, plug it here.
        explanation = {
            'summary': 'SmartSched selected SMART_HYBRID because predicted bursts are heterogeneous and RL confidence > 75%.',
            'details': {
                'predicted_variance': 12.3,
                'rl_confidence': 87
            }
        }
        return jsonify({'success': True, 'explanation': explanation})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/export', methods=['POST'])
def export_report():
    """Return a JSON file of the last run results (simple example). Extend to PDF via ReportLab/WeasyPrint."""
    try:
        data = request.get_json(force=True)
        # Build JSON report
        report = {
            'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'payload': data
        }
        bio = json.dumps(report, indent=2)
        buf = io.BytesIO()
        buf.write(bio.encode('utf-8'))
        buf.seek(0)
        return send_file(buf, as_attachment=True, attachment_filename='smartsched_report.json', mimetype='application/json')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("Starting SmartSched unified Flask app on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
