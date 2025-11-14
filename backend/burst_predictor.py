"""
burst_predictor.py - ML-based Burst Time Predictor
Uses Random Forest and Gradient Boosting for prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class BurstTimePredictor:
    """
    ML-based Burst Time Predictor for SmartSched
    Predicts process burst times based on historical patterns
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        
        self.feature_names = [
            'process_size',      # KB
            'priority',          # 1-10
            'arrival_time',      # Time units
            'prev_burst_avg',    # Average of previous bursts
            'process_type',      # 0:CPU, 1:IO, 2:Mixed, 3:Interactive
            'time_of_day',       # 0-23 hours
            'memory_usage',      # MB
            'cpu_affinity'       # 0-3 (core preference)
        ]
        
    def generate_synthetic_training_data(self, n_samples=10000):
        """
        Generate realistic synthetic process data for training
        Based on common OS process patterns
        """
        np.random.seed(42)
        
        # Generate features with realistic distributions
        data = {
            'process_size': np.random.lognormal(5, 1.5, n_samples).astype(int).clip(10, 2000),
            'priority': np.random.randint(1, 11, n_samples),
            'arrival_time': np.random.randint(0, 200, n_samples),
            'prev_burst_avg': np.random.lognormal(2.5, 0.8, n_samples).clip(1, 100),
            'process_type': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.2, 0.3, 0.2]),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'memory_usage': np.random.lognormal(4, 1, n_samples).astype(int).clip(10, 1000),
            'cpu_affinity': np.random.randint(0, 4, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic burst times based on features with non-linear relationships
        df['burst_time'] = (
            df['process_size'] * 0.03 +
            (11 - df['priority']) * 2.5 +  # Lower priority = longer burst
            df['prev_burst_avg'] * 0.7 +
            df['process_type'].map({0: 15, 1: 8, 2: 12, 3: 4}) +  # Type-specific base
            df['memory_usage'] * 0.02 +
            np.where(df['time_of_day'].between(9, 17), 5, 0) +  # Peak hours
            np.random.normal(0, 3, n_samples)  # Noise
        ).clip(1, 150).astype(int)
        
        # Add some complex interactions
        df.loc[df['process_type'] == 0, 'burst_time'] *= 1.3  # CPU-bound processes
        df.loc[df['process_type'] == 3, 'burst_time'] *= 0.6  # Interactive processes
        df.loc[df['priority'] >= 8, 'burst_time'] *= 0.8  # High priority = shorter
        
        return df
    
    def train(self, X=None, y=None, save_model=True, model_path='models/'):
        """
        Train the ML model on process data
        
        Args:
            X: Feature DataFrame (optional, will generate if None)
            y: Target Series (optional)
            save_model: Whether to save trained model
            model_path: Directory to save model
        """
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Generate synthetic data if none provided
        if X is None or y is None:
            print("📊 Generating synthetic training data...")
            df = self.generate_synthetic_training_data()
            X = df[self.feature_names]
            y = df['burst_time']
            print(f"   Generated {len(df)} training samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose and configure model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            print(f"🤖 Training Random Forest model...")
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
            print(f"🤖 Training Gradient Boosting model...")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                    cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        # Calculate accuracy percentage
        accuracy = (1 - mae/y_test.mean()) * 100
        
        print(f"\n✅ Model trained successfully!")
        print(f"   {'='*50}")
        print(f"   Mean Absolute Error (MAE):  {mae:.2f} time units")
        print(f"   Root Mean Squared Error:    {rmse:.2f} time units")
        print(f"   R² Score:                   {r2:.4f}")
        print(f"   Cross-Val MAE:              {cv_mae:.2f} time units")
        print(f"   Prediction Accuracy:        {accuracy:.1f}%")
        print(f"   {'='*50}")
        
        self.is_trained = True
        
        # Store training history
        self.training_history.append({
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy,
            'samples': len(X)
        })
        
        # Save model
        if save_model:
            model_file = os.path.join(model_path, f'{self.model_type}_model.pkl')
            scaler_file = os.path.join(model_path, 'scaler.pkl')
            
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"💾 Model saved to {model_file}")
            print(f"💾 Scaler saved to {scaler_file}")
        
        return mae, r2, accuracy
    
    def predict(self, process_features):
        """
        Predict burst time for a single process
        
        Args:
            process_features: dict or DataFrame with process attributes
        
        Returns:
            predicted_burst_time: int
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Convert to DataFrame if dict
        if isinstance(process_features, dict):
            process_features = pd.DataFrame([process_features])
        
        # Ensure correct feature order
        X = process_features[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        return max(1, int(round(prediction)))  # Ensure positive integer
    
    def predict_batch(self, processes_df):
        """
        Predict burst times for multiple processes
        
        Args:
            processes_df: DataFrame with process features
        
        Returns:
            predictions: numpy array of predicted burst times
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        X = processes_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return np.maximum(1, np.round(predictions).astype(int))
    
    def load_model(self, model_path='models/', model_type=None):
        """
        Load a pre-trained model
        
        Args:
            model_path: Directory containing model files
            model_type: Type of model to load (None = use self.model_type)
        """
        if model_type:
            self.model_type = model_type
        
        model_file = os.path.join(model_path, f'{self.model_type}_model.pkl')
        scaler_file = os.path.join(model_path, 'scaler.pkl')
        
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print(f"✅ Model loaded from {model_file}")
            return True
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {e}")
            return False
    
    def get_feature_importance(self):
        """
        Get feature importance for interpretation
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained or self.model_type not in ['random_forest']:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate model on a specific test set
        """
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        accuracy = (1 - mae/y_test.mean()) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy
        }


# Demo usage
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SmartSched ML Burst Time Predictor - Training Demo")
    print("=" * 70)
    
    # Initialize predictor
    predictor = BurstTimePredictor(model_type='random_forest')
    
    # Train model
    mae, r2, accuracy = predictor.train()
    
    print("\n" + "=" * 70)
    print("🧪 Testing Predictions on Sample Processes")
    print("=" * 70)
    
    # Test predictions
    test_processes = [
        {
            'process_size': 500,
            'priority': 3,
            'arrival_time': 0,
            'prev_burst_avg': 20,
            'process_type': 0,  # CPU-bound
            'time_of_day': 14,
            'memory_usage': 256,
            'cpu_affinity': 0
        },
        {
            'process_size': 100,
            'priority': 8,
            'arrival_time': 5,
            'prev_burst_avg': 8,
            'process_type': 3,  # Interactive
            'time_of_day': 9,
            'memory_usage': 64,
            'cpu_affinity': 1
        },
        {
            'process_size': 800,
            'priority': 2,
            'arrival_time': 10,
            'prev_burst_avg': 35,
            'process_type': 0,  # CPU-bound
            'time_of_day': 15,
            'memory_usage': 512,
            'cpu_affinity': 2
        }
    ]
    
    for i, proc in enumerate(test_processes, 1):
        predicted_burst = predictor.predict(proc)
        proc_type_name = ['CPU-Bound', 'I/O-Bound', 'Mixed', 'Interactive'][proc['process_type']]
        
        print(f"\n📌 Process {i} ({proc_type_name}):")
        print(f"   Size: {proc['process_size']}KB, Priority: {proc['priority']}, "
              f"Memory: {proc['memory_usage']}MB")
        print(f"   🎯 Predicted Burst Time: {predicted_burst} units")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("📊 Feature Importance Analysis")
    print("=" * 70)
    importance = predictor.get_feature_importance()
    if importance is not None:
        for idx, row in importance.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {row['feature']:<20} {bar} {row['importance']:.4f}")
    
    print("\n" + "=" * 70)
    print("✨ Model ready for SmartSched integration!")
    print("=" * 70)

    # ... (keep the entire existing file as is, but add this small fix at the end)

# Demo usage
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SmartSched ML Burst Time Predictor - Training Demo")
    print("=" * 70)
    
    # Initialize predictor
    predictor = BurstTimePredictor(model_type='random_forest')
    
    # Train model
    mae, r2, accuracy = predictor.train()
    
    # Rest of the demo code remains the same...
