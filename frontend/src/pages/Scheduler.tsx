import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Clock, Cpu, TrendingUp, Activity, Zap, Play } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import { socket } from "@/lib/socket";
import { useProcesses } from "@/contexts/ProcessContext";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

// --- TYPE DEFINITIONS ---
interface Metric {
  label: string;
  value: string;
  icon: React.ElementType;
  color: string;
}

interface GanttEntry {
  pid: number;
  start: number;
  duration: number;
  color: string;
}

interface ProcessMetric {
  pid: number;
  wait_time: number;
  turnaround_time: number;
  response_time: number;
}

// --- DEFAULT STATE ---
const initialMetrics: Metric[] = [
  { label: "Avg Wait Time", value: "0ms", icon: Clock, color: "text-primary" },
  { label: "Avg Response Time", value: "0ms", icon: Activity, color: "text-secondary" },
  { label: "CPU Utilization", value: "0%", icon: Cpu, color: "text-accent" },
  { label: "Throughput", value: "0/s", icon: TrendingUp, color: "text-success" },
  { label: "Context Switches", value: "0", icon: Zap, color: "text-primary" },
];

const Scheduler = () => {
  const [algorithm, setAlgorithm] = useState("ml-hybrid");
  const { processes } = useProcesses();
  
  // --- REAL-TIME STATE ---
  const [metrics, setMetrics] = useState<Metric[]>(initialMetrics);
  const [ganttData, setGanttData] = useState<GanttEntry[]>([]);
  const [processMetrics, setProcessMetrics] = useState<ProcessMetric[]>([]);
  const [totalTime, setTotalTime] = useState(25);

  const [isLoading, setIsLoading] = useState(false);

  // --- SOCKET.IO LISTENER ---
  useEffect(() => {
    function onSimulationComplete(data: { metrics: any }) {
      console.log("Simulation complete:", data.metrics);
      
      const { 
        avg_wait_time, 
        avg_response_time, 
        cpu_utilization, 
        throughput, 
        context_switches,
        gantt_chart,
        process_metrics,
        total_time
      } = data.metrics;

      // 1. Update Metrics Cards
      setMetrics([
        { label: "Avg Wait Time", value: `${avg_wait_time.toFixed(2)}ms`, icon: Clock, color: "text-primary" },
        { label: "Avg Response Time", value: `${avg_response_time.toFixed(2)}ms`, icon: Activity, color: "text-secondary" },
        { label: "CPU Utilization", value: `${cpu_utilization.toFixed(1)}%`, icon: Cpu, color: "text-accent" },
        { label: "Throughput", value: `${throughput.toFixed(2)}/s`, icon: TrendingUp, color: "text-success" },
        { label: "Context Switches", value: `${context_switches}`, icon: Zap, color: "text-primary" },
      ]);

      // 2. Update Gantt Chart
      if (gantt_chart) {
        const newGanttData = gantt_chart.map((item: any, index: number) => ({
          pid: item.pid,
          start: item.start,
          duration: item.end - item.start,
          color: `hsl(${(item.pid * 70) % 360}, 70%, 50%)`,
        }));
        setGanttData(newGanttData);
        setTotalTime(Math.max(25, total_time || 25));
      }

      // 3. Update Process Table Metrics
      if (process_metrics) {
        setProcessMetrics(process_metrics);
      }

      setIsLoading(false);
    }
    
    socket.on("simulation_complete", onSimulationComplete);

    return () => {
      socket.off("simulation_complete", onSimulationComplete);
    };
  }, []);

  // --- SIMULATION TRIGGER ---
  const runSimulation = () => {
    if (processes.length === 0) {
      alert("Please add processes on the 'Process Builder' page first.");
      return;
    }
    
    setIsLoading(true);
    setGanttData([]);
    setMetrics(initialMetrics);
    setProcessMetrics([]);
    
    socket.emit("run_simulation", { 
      processes: processes.map(p => ({...p, burst_time: p.burst_time || 10})),
      algorithm: algorithm
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="container mx-auto px-4 lg:px-8 pt-24 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-7xl mx-auto space-y-8"
        >
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">
                <span className="gradient-text">AI Scheduler</span>
              </h1>
              <p className="text-muted-foreground">
                Loaded {processes.length} processes from Builder. Ready to simulate.
              </p>
            </div>
            <Button size="lg" onClick={runSimulation} disabled={isLoading} className="w-48">
              {isLoading ? (
                "Simulating..."
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Simulation
                </>
              )}
            </Button>
          </div>

          {/* Metrics Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            {metrics.map((metric, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="glass border-border/50">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-2">
                      <metric.icon className={`h-5 w-5 ${metric.color}`} />
                    </div>
                    <div className="text-2xl font-bold">{metric.value}</div>
                    <div className="text-xs text-muted-foreground">{metric.label}</div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Main Content */}
          <Tabs defaultValue="gantt" className="space-y-6">
            <TabsList className="glass border border-border/50">
              <TabsTrigger value="gantt">Gantt Chart</TabsTrigger>
              <TabsTrigger value="table">Process Table</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            <TabsContent value="gantt" className="space-y-6">
              <Card className="glass border-border/50">
                <CardHeader>
                  <CardTitle>Execution Timeline</CardTitle>
                  <CardDescription>Process execution visualization with ML predictions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center space-x-2 text-sm text-muted-foreground mb-4">
                      {Array.from(new Set(ganttData.map(item => item.pid))).slice(0, 4).map(pid => (
                        <div key={pid} className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-1" 
                            style={{ backgroundColor: `hsl(${(pid * 70) % 360}, 70%, 50%)` }}
                          ></div>
                          <span>P{pid}</span>
                        </div>
                      ))}
                    </div>

                    {/* Timeline */}
                    <div className="relative h-64 glass-dark rounded-lg p-6 border border-border/50">
                      <div className="absolute bottom-6 left-6 right-6 h-2 bg-muted rounded-full"></div>
                      {ganttData.map((item, index) => (
                        <motion.div
                          key={index}
                          initial={{ width: 0 }}
                          animate={{ width: `${(item.duration / totalTime) * 100}%` }}
                          transition={{ delay: index * 0.2, duration: 0.5 }}
                          className="absolute h-12 rounded-lg shadow-lg border border-border/50 flex items-center justify-center font-semibold text-sm glow-primary"
                          style={{
                            left: `${(item.start / totalTime) * 100}%`,
                            bottom: "24px",
                            backgroundColor: `hsl(${(item.pid * 70) % 360}, 70%, 50%)`,
                          }}
                        >
                          P{item.pid}
                        </motion.div>
                      ))}
                      {/* Time markers */}
                      <div className="absolute bottom-2 left-6 right-6 flex justify-between text-xs text-muted-foreground">
                        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((perc) => (
                          <span key={perc}>{Math.round(perc * totalTime)}ms</span>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="table">
              <Card className="glass border-border/50">
                <CardHeader>
                  <CardTitle>Process Details</CardTitle>
                  <CardDescription>Comprehensive process information</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>PID</TableHead>
                        <TableHead>Wait Time</TableHead>
                        <TableHead>Response Time</TableHead>
                        <TableHead>Turnaround Time</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {processMetrics.length > 0 ? (
                        processMetrics.map((p) => (
                          <TableRow key={p.pid}>
                            <TableCell className="font-medium">P{p.pid}</TableCell>
                            <TableCell>{p.wait_time.toFixed(2)}ms</TableCell>
                            <TableCell>{p.response_time.toFixed(2)}ms</TableCell>
                            <TableCell>{p.turnaround_time.toFixed(2)}ms</TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={4} className="text-center h-24 text-muted-foreground">
                            Run a simulation to see process metrics.
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analytics">
              <Card className="glass border-border/50">
                <CardHeader>
                  <CardTitle>Performance Analytics</CardTitle>
                  <CardDescription>Deep insights and trends</CardDescription>
                </CardHeader>
                <CardContent className="h-[400px]">
                  {processMetrics.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={processMetrics}>
                        <XAxis 
                          dataKey="pid" 
                          stroke="#888888" 
                          fontSize={12} 
                          tickLine={false} 
                          axisLine={false} 
                          tickFormatter={(val) => `P${val}`} 
                        />
                        <YAxis 
                          stroke="#888888" 
                          fontSize={12} 
                          tickLine={false} 
                          axisLine={false} 
                        />
                        <Tooltip
                          contentStyle={{ 
                            backgroundColor: "hsl(var(--background))", 
                            borderColor: "hsl(var(--border))",
                            borderRadius: "var(--radius)"
                          }}
                        />
                        <Legend />
                        <Bar 
                          dataKey="wait_time" 
                          fill="hsl(var(--primary))" 
                          name="Wait Time" 
                          radius={[4, 4, 0, 0]} 
                        />
                        <Bar 
                          dataKey="response_time" 
                          fill="hsl(var(--secondary))" 
                          name="Response Time" 
                          radius={[4, 4, 0, 0]} 
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      Run a simulation to see analytics.
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Algorithm Selector */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Algorithm Selection</CardTitle>
              <CardDescription>Choose or let AI decide the optimal algorithm</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-4">
                {["ML-Hybrid", "FCFS", "SJF", "Priority"].map((algo) => (
                  <Button
                    key={algo}
                    variant={algorithm === algo.toLowerCase() ? "default" : "outline"}
                    className="h-auto py-4"
                    onClick={() => setAlgorithm(algo.toLowerCase())}
                    disabled={isLoading}
                  >
                    {algo}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Scheduler;
