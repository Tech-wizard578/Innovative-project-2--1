"""
lstm_predictor.py - LSTM-based Burst Time Predictor
Uses deep learning for time-series prediction of process bursts
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class LSTMBurstPredictor:
    """
    LSTM-based predictor for process burst times
    Uses time-series patterns to predict future burst durations
    """
    
    def __init__(self, sequence_length=10):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of past observations to use for prediction
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_trained = False
        self.history = None
        
        self.feature_names = [
            'process_size',
            'priority',
            'arrival_time',
            'prev_burst_1',
            'prev_burst_2',
            'prev_burst_3',
            'process_type',
            'time_of_day',
            'memory_usage',
            'cpu_load'
        ]
    
    def generate_time_series_data(self, n_sequences=1000):
        """
        Generate synthetic time-series process data
        Simulates realistic process burst patterns over time
        """
        np.random.seed(42)
        
        all_sequences = []
        all_targets = []
        
        for seq_id in range(n_sequences):
            # Generate a sequence of related processes
            seq_length = self.sequence_length + 10  # Extra for targets
            
            # Process characteristics (remain relatively stable in sequence)
            base_size = np.random.randint(100, 1000)
            base_priority = np.random.randint(1, 11)
            proc_type = np.random.choice([0, 1, 2, 3])
            
            sequence_data = []
            prev_bursts = [np.random.randint(5, 30) for _ in range(3)]
            
            for t in range(seq_length):
                # Features with temporal dependencies
                process_size = base_size + np.random.randint(-50, 50)
                priority = max(1, min(10, base_priority + np.random.randint(-1, 2)))
                arrival_time = t * 10 + np.random.randint(0, 5)
                time_of_day = (8 + t) % 24
                memory_usage = base_size * 0.5 + np.random.randint(0, 100)
                cpu_load = 50 + 30 * np.sin(t / 5) + np.random.randint(-10, 10)
                
                # Generate burst time with temporal correlation
                burst = (
                    process_size * 0.02 +
                    (11 - priority) * 2 +
                    np.mean(prev_bursts) * 0.6 +
                    proc_type * 3 +
                    memory_usage * 0.01 +
                    cpu_load * 0.1 +
                    np.random.normal(0, 2)
                )
                burst = max(1, int(burst))
                
                sequence_data.append([
                    process_size,
                    priority,
                    arrival_time,
                    prev_bursts[0],
                    prev_bursts[1],
                    prev_bursts[2],
                    proc_type,
                    time_of_day,
                    memory_usage,
                    cpu_load
                ])
                
                # Update previous bursts
                prev_bursts = [burst] + prev_bursts[:2]
            
            # Create sequences and targets
            for i in range(len(sequence_data) - self.sequence_length):
                seq = sequence_data[i:i + self.sequence_length]
                target = prev_bursts[0]  # Predict next burst
                
                all_sequences.append(seq)
                all_targets.append(target)
        
        return np.array(all_sequences), np.array(all_targets)
    
    def build_model(self, input_shape):
        """
        Build LSTM neural network architecture
        """
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X=None, y=None, epochs=50, batch_size=32, 
              save_model=True, model_path='models/'):
        """
        Train the LSTM model
        
        Args:
            X: Input sequences (optional, will generate if None)
            y: Target values (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_model: Whether to save the trained model
            model_path: Directory to save model
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Generate data if not provided
        if X is None or y is None:
            print("📊 Generating time-series training data...")
            X, y = self.generate_time_series_data(n_sequences=1000)
            print(f"   Generated {len(X)} sequences")
        
        # Reshape and scale data
        n_samples, seq_len, n_features = X.shape
        
        # Scale features
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Scale targets
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Split data
        split = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]
        
        print(f"\n🤖 Building LSTM model...")
        print(f"   Input shape: {X_train.shape[1:]}")
        
        # Build model
        self.model = self.build_model(input_shape=(seq_len, n_features))
        
        print(f"\n📊 Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(model_path, 'lstm_best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        print(f"\n🚀 Training LSTM model for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        accuracy = (1 - mae / y_test_orig.mean()) * 100
        
        print(f"\n✅ LSTM Model trained successfully!")
        print(f"   {'='*50}")
        print(f"   Mean Absolute Error:  {mae:.2f} time units")
        print(f"   R² Score:             {r2:.4f}")
        print(f"   Prediction Accuracy:  {accuracy:.1f}%")
        print(f"   {'='*50}")
        
        self.is_trained = True
        
        # Save model and scalers
        if save_model:
            model_file = os.path.join(model_path, 'lstm_model.h5')
            scaler_x_file = os.path.join(model_path, 'lstm_scaler_x.pkl')
            scaler_y_file = os.path.join(model_path, 'lstm_scaler_y.pkl')
            
            self.model.save(model_file)
            with open(scaler_x_file, 'wb') as f:
                pickle.dump(self.scaler_X, f)
            with open(scaler_y_file, 'wb') as f:
                pickle.dump(self.scaler_y, f)
            
            print(f"💾 Model saved to {model_file}")
        
        return mae, r2, accuracy
    
    def predict(self, sequence):
        """
        Predict burst time for next process given a sequence
        
        Args:
            sequence: Array of shape (sequence_length, n_features)
        
        Returns:
            predicted_burst_time: int
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        # Scale input
        n_samples, seq_len, n_features = sequence.shape
        seq_reshaped = sequence.reshape(-1, n_features)
        seq_scaled = self.scaler_X.transform(seq_reshaped)
        seq_scaled = seq_scaled.reshape(n_samples, seq_len, n_features)
        
        # Predict
        pred_scaled = self.model.predict(seq_scaled, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)
        
        return max(1, int(round(pred[0][0])))
    
    def load_model(self, model_path='models/'):
        """
        Load pre-trained LSTM model
        """
        model_file = os.path.join(model_path, 'lstm_model.h5')
        scaler_x_file = os.path.join(model_path, 'lstm_scaler_x.pkl')
        scaler_y_file = os.path.join(model_path, 'lstm_scaler_y.pkl')
        
        try:
            self.model = load_model(model_file)
            with open(scaler_x_file, 'rb') as f:
                self.scaler_X = pickle.load(f)
            with open(scaler_y_file, 'rb') as f:
                self.scaler_y = pickle.load(f)
            
            self.is_trained = True
            print(f"✅ LSTM model loaded from {model_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SmartSched LSTM Burst Time Predictor - Training Demo")
    print("=" * 70)
    
    # Initialize predictor
    lstm_predictor = LSTMBurstPredictor(sequence_length=10)
    
    # Train model (use fewer epochs for demo)
    mae, r2, accuracy = lstm_predictor.train(epochs=30, batch_size=32)
    
    print("\n" + "=" * 70)
    print("✨ LSTM Model ready for time-series burst prediction!")
    print("=" * 70)