import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Cpu, Clock, TrendingUp } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";

// --- NEW IMPORTS ---
import { useEffect, useState } from "react";
import { socket } from "@/lib/socket"; // Import your new socket client
// --- END NEW IMPORTS ---

const Dashboard = () => {
  // --- STATE FOR LIVE DATA ---
  // Replace the 'liveMetrics' mock array with state
  const [metrics, setMetrics] = useState([
    { label: "Active Processes", value: "0", change: "+0", icon: Activity },
    { label: "CPU Load", value: "0%", change: "0%", icon: Cpu },
    { label: "Avg Wait", value: "0ms", change: "0ms", icon: Clock },
    { label: "Throughput", value: "0/s", change: "+0", icon: TrendingUp },
  ]);

  const [liveLogs, setLiveLogs] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(socket.connected);
  // --- END STATE ---

  // --- EFFECT FOR SOCKETS ---
  useEffect(() => {
    // Listen for connection
    function onConnect() {
      setIsConnected(true);
      console.log("Socket connected!");
    }

    // Listen for disconnection
    function onDisconnect() {
      setIsConnected(false);
      console.log("Socket disconnected.");
    }

    // Listen for 'process_log' events from your server
    function onProcessLog(data: { message: string }) {
      setLiveLogs((prevLogs) => [...prevLogs, data.message]); // Add new log to the bottom
    }

    // Listen for 'simulation_complete' events
    function onSimComplete(data: { metrics: any }) {
      // This is where you would update the 'metrics' state
      console.log("Simulation complete:", data.metrics);
      // Example:
      // setMetrics([ ... new metrics data ... ]);
    }

    // Register event listeners
    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("process_log", onProcessLog);
    socket.on("simulation_complete", onSimComplete);

    // Clean up listeners on component unmount
    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("process_log", onProcessLog);
      socket.off("simulation_complete", onSimComplete);
    };
  }, []);
  // --- END EFFECT ---

  // --- FUNCTION TO START SIMULATION ---
  const startSimulation = () => {
    // This sends the 'run_simulation' message to your Python server
    console.log("Requesting simulation run...");
    setLiveLogs([]); // Clear logs
    socket.emit("run_simulation", {
      // You can send data here, e.g., processes from ProcessBuilder
      processes: [
        { pid: 1, arrival_time: 0, burst_time: 8, priority: 3 },
        { pid: 2, arrival_time: 1, burst_time: 4, priority: 1 },
        { pid: 3, arrival_time: 2, burst_time: 5, priority: 2 },
      ],
    });
  };
  // --- END FUNCTION ---

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="container mx-auto px-4 lg:px-8 pt-24 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-7xl mx-auto space-y-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">
                <span className="gradient-text">Real-Time Dashboard</span>
              </h1>
              <p className="text-muted-foreground">Live monitoring of CPU scheduling</p>
            </div>
            {/* --- MODIFIED BADGE --- */}
            <Badge
              className={`animate-pulse border-success/50 ${
                isConnected ? "bg-success/20 text-success" : "bg-destructive/20 text-destructive"
              }`}
            >
              ● {isConnected ? "Live" : "Offline"}
            </Badge>
            {/* --- END MODIFIED BADGE --- */}
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* --- MODIFIED METRICS MAPPING --- */}
            {metrics.map((metric, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="glass border-border/50">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-3">
                      <metric.icon className="h-5 w-5 text-primary" />
                      <Badge variant="outline" className="text-xs">
                        {metric.change}
                      </Badge>
                    </div>
                    <div className="text-3xl font-bold mb-1">{metric.value}</div>
                    <div className="text-sm text-muted-foreground">{metric.label}</div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
            {/* --- END MODIFIED METRICS --- */}
          </div>

          <Card className="glass border-border/50">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Live Process Queue</CardTitle>
                <CardDescription>Real-time process execution stream</CardDescription>
              </div>
              {/* --- ADD THIS BUTTON --- */}
              <button
                onClick={startSimulation}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md shadow hover:bg-primary/90"
              >
                Run Simulation
              </button>
              {/* --- END BUTTON --- */}
            </CardHeader>
            <CardContent>
              {/* --- MODIFIED CONTENT --- */}
              {/* This is your new terminal */}
              <div className="h-64 overflow-y-auto bg-muted/50 rounded-lg p-4 font-mono text-sm border border-border/50">
                {liveLogs.length > 0 ? (
                  liveLogs.map((log, index) => (
                    <p key={index}>{log}</p>
                  ))
                ) : (
                  <p className="text-muted-foreground">
                    {isConnected ? "Socket.IO connected. Waiting for logs..." : "Connecting to backend..."}
                  </p>
                )}
              </div>
              {/* --- END MODIFIED CONTENT --- */}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
