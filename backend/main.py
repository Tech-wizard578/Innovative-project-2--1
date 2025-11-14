import asyncio
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# --- 1. Import Your AI Logic ---
from burst_predictor import BurstTimePredictor
from lstm_predictor import LSTMBurstPredictor
from multicore_scheduler import MultiCoreScheduler, Process

print("🚀 Starting SmartSched Backend...")

# --- 2. Initialize Server and Models ---
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Add CORS middleware to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load Your Models on Startup ---
print("🧠 Loading AI models...")
burst_predictor_rf = BurstTimePredictor(model_type='random_forest')

# Train the model if it doesn't exist, otherwise load it
try:
    # Try to load existing model
    if not burst_predictor_rf.load_model(model_path="models/"):
        # If loading fails, train a new model
        print("🔄 Training new Random Forest model...")
        burst_predictor_rf.train(save_model=True, model_path="models/")
except Exception as e:
    print(f"⚠️  Error with Random Forest model: {e}")
    # Fallback: train new model
    burst_predictor_rf.train(save_model=True, model_path="models/")

lstm_predictor = LSTMBurstPredictor(sequence_length=10)
try:
    if not lstm_predictor.load_model(model_path="models/"):
        print("🔄 Training new LSTM model...")
        lstm_predictor.train(epochs=20, save_model=True, model_path="models/")
except Exception as e:
    print(f"⚠️  Error with LSTM model: {e}")
    # Fallback: train new model
    lstm_predictor.train(epochs=20, save_model=True, model_path="models/")

print("✅ Models loaded successfully.")

# --- 4. Create API Endpoints ---

class ProcessInput(BaseModel):
    process_size: int
    priority: int
    arrival_time: int
    prev_burst_avg: int
    process_type: int
    time_of_day: int
    memory_usage: int
    cpu_affinity: int

@app.post("/api/predict/burst-time")
async def predict_burst(process_input: ProcessInput):
    """
    API endpoint to predict the burst time for a single process.
    """
    try:
        input_dict = process_input.dict()
        predicted_burst = burst_predictor_rf.predict(input_dict)
        
        return {
            "predicted_burst_time": predicted_burst,
            "input_data": input_dict
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/api/metrics/default")
async def get_default_metrics():
    """
    A simple endpoint to send mock metrics.
    """
    return [
        {"label": "Avg Wait Time", "value": "0ms", "icon": "Clock"},
        {"label": "CPU Utilization", "value": "0%", "icon": "Cpu"},
        {"label": "Throughput", "value": "0/s", "icon": "TrendingUp"},
    ]

# --- 5. Create WebSocket Events ---

@sio.on('connect')
async def connect(sid, environ):
    """
    This event fires when your React app connects to the WebSocket.
    """
    print(f"🔗 Client connected: {sid}")
    await sio.emit('process_log', {'message': 'SmartSched Backend Connected!'})

@sio.on('run_simulation')
async def run_simulation(sid, data):
    """
    This event fires when your React app sends a 'run_simulation' message.
    """
    print("🏃‍♂️ Simulation run requested...")
    await sio.emit('process_log', {'message': 'Simulation starting...'})
    
    try:
        processes_data = data.get('processes', [])
        if not processes_data:
            # Create default processes if none are provided
            processes_data = [
                {'pid': 1, 'arrival_time': 0, 'burst_time': 8, 'priority': 3},
                {'pid': 2, 'arrival_time': 1, 'burst_time': 4, 'priority': 1},
                {'pid': 3, 'arrival_time': 2, 'burst_time': 5, 'priority': 2},
            ]

        # Convert dicts to Process objects
        processes = [Process(
            pid=p['pid'], 
            arrival_time=p['arrival_time'], 
            burst_time=p['burst_time'],
            priority=p.get('priority', 5),
            process_type=p.get('process_type', 0)
        ) for p in processes_data]

        scheduler = MultiCoreScheduler(num_cores=2)
        
        # Define a log function to send logs over WebSocket
        async def async_log(message):
            print(message)
            await sio.emit('process_log', {'message': message})

        scheduler.log_function = async_log
        
        # Run the actual scheduling simulation - USE THE CORRECT FUNCTION NAME
        metrics = await scheduler.run_simulation(processes)
        
        await sio.emit('process_log', {'message': '✅ Simulation completed successfully!'})
        await sio.emit('simulation_complete', {'metrics': metrics})

    except Exception as e:
        print(f"❌ Simulation Error: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('process_log', {'message': f"Error: {e}"})
        # Send minimal response to unstick the frontend
        await sio.emit('simulation_complete', {'metrics': {
            "avg_wait_time": 0,
            "avg_response_time": 0,
            "cpu_utilization": 0,
            "throughput": 0,
            "context_switches": 0,
            "gantt_chart": [],
            "process_metrics": [],
            "total_time": 0
        }})



# --- 6. Run the Server ---
if __name__ == "__main__":
    print("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run(
        "main:socket_app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
