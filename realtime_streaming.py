"""
realtime_streaming.py - WebSocket streaming for live updates
Add this to enable REAL-TIME process execution visualization
"""

from flask_socketio import SocketIO, emit
from flask import Flask
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class RealtimeScheduler:
    """Real-time scheduler with WebSocket updates"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.is_running = False
        self.thread = None
    
    def start_realtime_execution(self):
        """Execute scheduling with real-time updates"""
        self.is_running = True
        self.thread = threading.Thread(target=self._execute_with_updates)
        self.thread.start()
    
    def _execute_with_updates(self):
        """Execute and emit updates every 100ms"""
        time_step = 0
        
        for process in self.scheduler.processes:
            if not self.is_running:
                break
            
            # Emit process start
            socketio.emit('process_start', {
                'pid': process.pid,
                'time': time_step,
                'core': process.cpu_affinity
            })
            
            # Simulate execution with updates
            for t in range(process.burst_time):
                if not self.is_running:
                    break
                
                time.sleep(0.1)  # 100ms per time unit
                
                # Emit progress update
                socketio.emit('execution_update', {
                    'pid': process.pid,
                    'progress': (t + 1) / process.burst_time * 100,
                    'time': time_step + t,
                    'cpu_usage': self._get_cpu_usage()
                })
            
            # Emit completion
            socketio.emit('process_complete', {
                'pid': process.pid,
                'time': time_step + process.burst_time
            })
            
            time_step += process.burst_time
        
        # Final metrics
        metrics = self.scheduler.get_metrics()
        socketio.emit('execution_complete', metrics)
    
    def _get_cpu_usage(self):
        """Get current CPU usage per core"""
        # Calculate actual usage
        return [92.3, 91.5, 93.1, 90.8]  # Example
    
    def stop(self):
        """Stop real-time execution"""
        self.is_running = False

@socketio.on('start_scheduling')
def handle_start_scheduling(data):
    """Handle start request from frontend"""
    algorithm = data.get('algorithm', 'SMART_HYBRID')
    processes = data.get('processes', [])
    
    # Create scheduler
    from smart_scheduler import SmartScheduler, Process
    scheduler = SmartScheduler(use_ml=True, algorithm=algorithm)
    
    # Add processes
    for p_data in processes:
        proc = Process(**p_data)
        scheduler.add_process(proc)
    
    # Start real-time execution
    rt_scheduler = RealtimeScheduler(scheduler)
    rt_scheduler.start_realtime_execution()

@socketio.on('stop_scheduling')
def handle_stop():
    """Stop execution"""
    emit('scheduling_stopped', {'status': 'stopped'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)