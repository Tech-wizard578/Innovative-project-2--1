// src/contexts/ProcessContext.tsx
import React, { createContext, useState, useContext, ReactNode } from "react";

// This defines the structure of a single process
export interface Process {
  pid: number;
  arrival_time: number;
  burst_time: number;
  priority: number;
  process_type: number; // 0:CPU, 1:IO, 2:Mixed, 3:Interactive
  // Add other fields from your builder form
}

// Define what the context will provide
interface ProcessContextType {
  processes: Process[];
  addProcess: (process: Omit<Process, "pid">) => void;
  clearProcesses: () => void;
}

const ProcessContext = createContext<ProcessContextType | undefined>(undefined);

export const ProcessProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [processes, setProcesses] = useState<Process[]>([]);

  const addProcess = (process: Omit<Process, "pid">) => {
    setProcesses((prev) => [
      ...prev,
      { ...process, pid: prev.length + 1 }, // Assign a new PID
    ]);
  };

  const clearProcesses = () => {
    setProcesses([]);
  };

  return (
    <ProcessContext.Provider value={{ processes, addProcess, clearProcesses }}>
      {children}
    </ProcessContext.Provider>
  );
};

// Custom hook to easily use the context
export const useProcesses = () => {
  const context = useContext(ProcessContext);
  if (!context) {
    throw new Error("useProcesses must be used within a ProcessProvider");
  }
  return context;
};