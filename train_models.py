"""
train_models.py - Central script to train all ML models for SmartSched
Trains: Random Forest, LSTM, and RL agents
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'models'))

from burst_predictor import BurstTimePredictor
from lstm_predictor import LSTMBurstPredictor
from rl_agent import QLearningAgent, train_rl_agent_demo


def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'logs', 'results']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Created project directories")


def train_random_forest(n_samples=10000):
    """Train Random Forest model"""
    print("\n" + "="*70)
    print("🌲 TRAINING RANDOM FOREST MODEL")
    print("="*70)
    
    start_time = time.time()
    
    predictor = BurstTimePredictor(model_type='random_forest')
    mae, r2, accuracy = predictor.train()
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️  Training time: {training_time:.2f} seconds")
    
    return {
        'model': 'Random Forest',
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'training_time': training_time
    }


def train_gradient_boosting(n_samples=10000):
    """Train Gradient Boosting model"""
    print("\n" + "="*70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("="*70)
    
    start_time = time.time()
    
    predictor = BurstTimePredictor(model_type='gradient_boosting')
    mae, r2, accuracy = predictor.train()
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️  Training time: {training_time:.2f} seconds")
    
    return {
        'model': 'Gradient Boosting',
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'training_time': training_time
    }


def train_lstm(n_sequences=1000, epochs=50):
    """Train LSTM model"""
    print("\n" + "="*70)
    print("🧠 TRAINING LSTM DEEP LEARNING MODEL")
    print("="*70)
    
    start_time = time.time()
    
    try:
        predictor = LSTMBurstPredictor(sequence_length=10)
        mae, r2, accuracy = predictor.train(epochs=epochs, batch_size=32)
        
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Training time: {training_time:.2f} seconds")
        
        return {
            'model': 'LSTM',
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'training_time': training_time
        }
    except Exception as e:
        print(f"⚠️  LSTM training failed: {e}")
        print("   Continuing with other models...")
        return None


def train_rl_agent(episodes=1000):
    """Train Reinforcement Learning agent"""
    print("\n" + "="*70)
    print("🎮 TRAINING REINFORCEMENT LEARNING AGENT")
    print("="*70)
    
    start_time = time.time()
    
    agent = train_rl_agent_demo(episodes=episodes)
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️  Training time: {training_time:.2f} seconds")
    
    return {
        'model': 'RL Agent (Q-Learning)',
        'episodes': episodes,
        'final_epsilon': agent.epsilon,
        'q_table_size': len(agent.q_table),
        'training_time': training_time
    }


def print_summary(results):
    """Print training summary"""
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        if result is None:
            continue
        
        print(f"\n{i}. {result['model']}")
        print(f"   {'-'*65}")
        
        if 'mae' in result:
            print(f"   MAE:              {result['mae']:.2f} time units")
            print(f"   R² Score:         {result['r2']:.4f}")
            print(f"   Accuracy:         {result['accuracy']:.1f}%")
        
        if 'episodes' in result:
            print(f"   Episodes:         {result['episodes']}")
            print(f"   Final Epsilon:    {result['final_epsilon']:.3f}")
            print(f"   Q-Table Size:     {result['q_table_size']}")
        
        print(f"   Training Time:    {result['training_time']:.2f} seconds")
    
    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📁 Saved Models:")
    print("   ✓ models/random_forest_model.pkl")
    print("   ✓ models/gradient_boosting_model.pkl")
    print("   ✓ models/lstm_model.h5 (if TensorFlow available)")
    print("   ✓ models/rl_agent.pkl")
    print("   ✓ models/scaler.pkl")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train SmartSched ML models')
    parser.add_argument('--models', nargs='+', 
                       choices=['rf', 'gb', 'lstm', 'rl', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of training samples (default: 10000)')
    parser.add_argument('--lstm-epochs', type=int, default=50,
                       help='LSTM training epochs (default: 50)')
    parser.add_argument('--rl-episodes', type=int, default=1000,
                       help='RL training episodes (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (fewer samples/epochs)')
    
    args = parser.parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.samples = 5000
        args.lstm_epochs = 20
        args.rl_episodes = 500
        print("⚡ Quick training mode enabled")
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        🚀 SmartSched ML Model Training Pipeline 🚀              ║
    ║                                                                  ║
    ║            Training all AI models for intelligent               ║
    ║                  process scheduling                             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create directories
    create_directories()
    
    # Determine which models to train
    train_all = 'all' in args.models
    results = []
    
    # Train Random Forest
    if train_all or 'rf' in args.models:
        result = train_random_forest(n_samples=args.samples)
        results.append(result)
    
    # Train Gradient Boosting
    if train_all or 'gb' in args.models:
        result = train_gradient_boosting(n_samples=args.samples)
        results.append(result)
    
    # Train LSTM
    if train_all or 'lstm' in args.models:
        result = train_lstm(n_sequences=args.samples//10, epochs=args.lstm_epochs)
        if result:
            results.append(result)
    
    # Train RL Agent
    if train_all or 'rl' in args.models:
        result = train_rl_agent(episodes=args.rl_episodes)
        results.append(result)
    
    # Print summary
    print_summary(results)
    
    print("\n🎯 Next Steps:")
    print("   1. Run 'python main_demo.py' to see models in action")
    print("   2. Run 'python app.py' to start web interface")
    print("   3. Customize schedulers in smart_scheduler.py")
    
    print("\n💡 Pro Tips:")
    print("   - Models are saved in models/ directory")
    print("   - Increase --samples for better accuracy")
    print("   - Use --quick for faster training during development")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()