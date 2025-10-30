"""
data_loader.py - Load and Process Google Borg Cluster Traces
Place in: D:\SmartSched\data_loader.py
"""

import pandas as pd
import numpy as np
import os


class BorgDataLoader:
    """
    Load and process Google Borg cluster trace data
    Converts real datacenter traces into process scheduling format
    """
    
    def __init__(self, data_path='data/borg_traces_data.csv'):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
    
    def load_data(self, max_rows=10000):
        """
        Load Borg trace data
        
        Args:
            max_rows: Maximum rows to load (for memory efficiency)
        """
        print(f"\n📂 Loading Google Borg trace data from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"❌ File not found: {self.data_path}")
            return None
        
        try:
            # Load data
            self.raw_data = pd.read_csv(self.data_path, nrows=max_rows)
            print(f"✅ Loaded {len(self.raw_data)} records from Borg traces")
            
            # Show columns
            print(f"\n📋 Available columns:")
            for i, col in enumerate(self.raw_data.columns, 1):
                print(f"   {i}. {col}")
            
            return self.raw_data
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def process_for_scheduling(self):
        """
        Convert Borg traces to scheduling format
        
        Borg traces typically have:
        - timestamp, task_id, job_id, priority
        - cpu_request, memory_request
        - execution time, start_time, end_time
        """
        if self.raw_data is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
        
        print("\n⚙️  Processing Borg data for scheduling simulation...")
        
        df = self.raw_data.copy()
        
        # Identify columns (flexible - works with different Borg formats)
        col_mapping = self._identify_columns(df)
        
        if not col_mapping:
            print("⚠️  Could not identify required columns. Using generic processing.")
            return self._generic_processing(df)
        
        # Extract and convert to process format
        processes = []
        
        for idx, row in df.iterrows():
            if idx >= 5000:  # Limit for performance
                break
            
            try:
                process = {
                    'pid': idx + 1,
                    'arrival_time': self._extract_arrival_time(row, col_mapping, idx),
                    'burst_time': self._extract_burst_time(row, col_mapping),
                    'priority': self._extract_priority(row, col_mapping),
                    'process_size': self._extract_cpu_request(row, col_mapping),
                    'memory_usage': self._extract_memory_request(row, col_mapping),
                    'process_type': self._determine_process_type(row, col_mapping),
                    'cpu_affinity': 0
                }
                
                # Validate
                if process['burst_time'] > 0 and process['burst_time'] < 200:
                    processes.append(process)
            
            except Exception as e:
                continue
        
        self.processed_data = pd.DataFrame(processes)
        
        print(f"✅ Processed {len(self.processed_data)} valid processes")
        print(f"\n📊 Data Statistics:")
        print(f"   Avg Burst Time:    {self.processed_data['burst_time'].mean():.2f}")
        print(f"   Avg Priority:      {self.processed_data['priority'].mean():.2f}")
        print(f"   Avg Process Size:  {self.processed_data['process_size'].mean():.2f}")
        
        return self.processed_data
    
    def _identify_columns(self, df):
        """Identify relevant columns in Borg data"""
        col_mapping = {}
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Try to find common Borg column names
        time_cols = ['timestamp', 'start_time', 'submit_time', 'time']
        duration_cols = ['duration', 'runtime', 'execution_time', 'cpu_time']
        priority_cols = ['priority', 'scheduling_class']
        cpu_cols = ['cpu', 'cpu_request', 'cpu_usage', 'cpus_requested']
        memory_cols = ['memory', 'mem_request', 'memory_usage', 'mem']
        
        for col_list, key in [
            (time_cols, 'time'),
            (duration_cols, 'duration'),
            (priority_cols, 'priority'),
            (cpu_cols, 'cpu'),
            (memory_cols, 'memory')
        ]:
            for col_name in col_list:
                if col_name in columns_lower:
                    col_mapping[key] = columns_lower[col_name]
                    break
        
        return col_mapping
    
    def _extract_arrival_time(self, row, col_mapping, idx):
        """Extract arrival time from row"""
        if 'time' in col_mapping:
            val = row[col_mapping['time']]
            if pd.notna(val):
                # Normalize to small numbers for simulation
                return int((val % 100) / 10)
        return idx % 20
    
    def _extract_burst_time(self, row, col_mapping):
        """Extract burst time from row"""
        if 'duration' in col_mapping:
            val = row[col_mapping['duration']]
            if pd.notna(val) and val > 0:
                # Scale to reasonable range (1-100)
                return max(1, min(100, int(val / 1000) if val > 1000 else int(val)))
        
        # Fallback: use CPU request as proxy
        if 'cpu' in col_mapping:
            val = row[col_mapping['cpu']]
            if pd.notna(val) and val > 0:
                return max(1, min(100, int(val * 20)))
        
        return np.random.randint(5, 50)
    
    def _extract_priority(self, row, col_mapping):
        """Extract priority from row"""
        if 'priority' in col_mapping:
            val = row[col_mapping['priority']]
            if pd.notna(val):
                # Normalize to 1-10 scale
                if val < 0:
                    return 10  # Free tier = low priority
                elif val > 10:
                    return max(1, int(val / 10))
                else:
                    return int(val)
        return np.random.randint(1, 11)
    
    def _extract_cpu_request(self, row, col_mapping):
        """Extract CPU request as process size"""
        if 'cpu' in col_mapping:
            val = row[col_mapping['cpu']]
            if pd.notna(val) and val > 0:
                # Scale to KB (100-2000)
                return max(100, min(2000, int(val * 1000)))
        return np.random.randint(100, 1000)
    
    def _extract_memory_request(self, row, col_mapping):
        """Extract memory request"""
        if 'memory' in col_mapping:
            val = row[col_mapping['memory']]
            if pd.notna(val) and val > 0:
                # Scale to MB (64-1024)
                return max(64, min(1024, int(val * 1000)))
        return np.random.randint(64, 512)
    
    def _determine_process_type(self, row, col_mapping):
        """Determine process type based on characteristics"""
        # 0: CPU-bound, 1: I/O-bound, 2: Mixed, 3: Interactive
        
        if 'cpu' in col_mapping and 'duration' in col_mapping:
            cpu = row[col_mapping['cpu']]
            duration = row[col_mapping['duration']]
            
            if pd.notna(cpu) and pd.notna(duration):
                if cpu > 0.8 and duration > 1000:
                    return 0  # CPU-bound
                elif cpu < 0.3 and duration < 100:
                    return 3  # Interactive
                elif cpu < 0.5:
                    return 1  # I/O-bound
                else:
                    return 2  # Mixed
        
        return np.random.choice([0, 1, 2, 3], p=[0.3, 0.2, 0.3, 0.2])
    
    def _generic_processing(self, df):
        """Generic processing when columns can't be identified"""
        print("⚙️  Using generic processing...")
        
        processes = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for idx in range(min(5000, len(df))):
            row = df.iloc[idx]
            
            # Use first few numeric columns as features
            values = [row[col] for col in numeric_cols[:5] if pd.notna(row[col])]
            
            if len(values) >= 2:
                process = {
                    'pid': idx + 1,
                    'arrival_time': idx % 20,
                    'burst_time': max(1, min(100, int(abs(values[0]) % 100 + 1))),
                    'priority': max(1, min(10, int(abs(values[1]) % 10 + 1))),
                    'process_size': max(100, min(2000, int(abs(values[0]) * 10))),
                    'memory_usage': 128,
                    'process_type': idx % 4,
                    'cpu_affinity': 0
                }
                processes.append(process)
        
        self.processed_data = pd.DataFrame(processes)
        return self.processed_data
    
    def get_training_data(self):
        """Get data in format for ML training"""
        if self.processed_data is None:
            self.process_for_scheduling()
        
        # Add time_of_day feature
        df = self.processed_data.copy()
        df['time_of_day'] = (df['arrival_time'] % 24)
        df['prev_burst_avg'] = df['burst_time'].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def save_processed_data(self, output_path='data/processed_borg_data.csv'):
        """Save processed data"""
        if self.processed_data is None:
            print("❌ No processed data to save")
            return
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"💾 Saved processed data to {output_path}")


# Demo usage
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Google Borg Data Loader - Demo")
    print("=" * 70)
    
    loader = BorgDataLoader('data/borg_traces_data.csv')
    
    # Load raw data
    raw_data = loader.load_data(max_rows=10000)
    
    if raw_data is not None:
        # Process for scheduling
        processed = loader.process_for_scheduling()
        
        if processed is not None:
            print("\n📋 Sample Processes:")
            print(processed.head(10))
            
            # Save
            loader.save_processed_data()
            
            print("\n✅ Borg data ready for SmartSched!")