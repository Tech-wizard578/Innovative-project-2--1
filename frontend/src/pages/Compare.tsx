import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Trophy, Zap, Clock } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const Compare = () => {
  // Mock comparison data
  const radarData = [
    { metric: "Wait Time", FCFS: 65, SJF: 85, Priority: 75, "ML-Hybrid": 95 },
    { metric: "Response Time", FCFS: 60, SJF: 80, Priority: 70, "ML-Hybrid": 92 },
    { metric: "CPU Util", FCFS: 70, SJF: 75, Priority: 80, "ML-Hybrid": 94 },
    { metric: "Throughput", FCFS: 65, SJF: 78, Priority: 82, "ML-Hybrid": 96 },
    { metric: "Fairness", FCFS: 90, SJF: 60, Priority: 70, "ML-Hybrid": 88 },
  ];

  const energyData = [
    { name: "FCFS", energy: 45, predicted: 42 },
    { name: "SJF", energy: 38, predicted: 36 },
    { name: "Priority", energy: 42, predicted: 40 },
    { name: "ML-Hybrid", energy: 28, predicted: 27 },
  ];

  const algorithms = [
    {
      name: "ML-Hybrid",
      avgWait: "2.4ms",
      avgResponse: "3.1ms",
      cpuUtil: "94%",
      badges: ["BEST PERFORMANCE", "MOST ENERGY EFFICIENT"],
      rank: 1,
    },
    {
      name: "Priority",
      avgWait: "3.8ms",
      avgResponse: "4.2ms",
      cpuUtil: "89%",
      badges: ["GOOD BALANCE"],
      rank: 2,
    },
    {
      name: "SJF",
      avgWait: "4.2ms",
      avgResponse: "5.1ms",
      cpuUtil: "87%",
      badges: ["LOW WAIT TIME"],
      rank: 3,
    },
    {
      name: "FCFS",
      avgWait: "6.5ms",
      avgResponse: "7.2ms",
      cpuUtil: "82%",
      badges: [],
      rank: 4,
    },
  ];

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
          <div>
            <h1 className="text-4xl font-bold mb-2">
              <span className="gradient-text">Algorithm Comparison</span>
            </h1>
            <p className="text-muted-foreground">
              Compare scheduling algorithms across multiple performance metrics
            </p>
          </div>

          {/* Performance Comparison Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {algorithms.map((algo, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card
                  className={`glass border-border/50 ${
                    algo.rank === 1 ? "border-primary/50 glow-primary" : ""
                  }`}
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{algo.name}</CardTitle>
                      {algo.rank === 1 && <Trophy className="h-5 w-5 text-primary" />}
                    </div>
                    <CardDescription>Rank #{algo.rank}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Wait:</span>
                        <span className="font-semibold">{algo.avgWait}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Response:</span>
                        <span className="font-semibold">{algo.avgResponse}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">CPU Util:</span>
                        <span className="font-semibold">{algo.cpuUtil}</span>
                      </div>
                    </div>
                    {algo.badges.length > 0 && (
                      <div className="flex flex-wrap gap-2 pt-2">
                        {algo.badges.map((badge, i) => (
                          <Badge key={i} className="text-xs bg-primary/20 text-primary border-primary/50">
                            {badge}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Radar Chart */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Multi-Metric Performance Analysis</CardTitle>
              <CardDescription>Normalized scores across key performance indicators</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "hsl(var(--foreground))" }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "hsl(var(--muted-foreground))" }} />
                  <Radar
                    name="FCFS"
                    dataKey="FCFS"
                    stroke="hsl(var(--muted))"
                    fill="hsl(var(--muted))"
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="SJF"
                    dataKey="SJF"
                    stroke="hsl(var(--accent))"
                    fill="hsl(var(--accent))"
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="Priority"
                    dataKey="Priority"
                    stroke="hsl(var(--secondary))"
                    fill="hsl(var(--secondary))"
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="ML-Hybrid"
                    dataKey="ML-Hybrid"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Energy Comparison */}
          <Card className="glass border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Energy Efficiency Comparison</CardTitle>
                  <CardDescription>Power consumption in watts (actual vs predicted)</CardDescription>
                </div>
                <Zap className="h-8 w-8 text-success" />
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={energyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" tick={{ fill: "hsl(var(--foreground))" }} />
                  <YAxis tick={{ fill: "hsl(var(--muted-foreground))" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Bar dataKey="energy" fill="hsl(var(--primary))" name="Actual Energy (W)" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="predicted" fill="hsl(var(--secondary))" name="ML Predicted (W)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Key Insights */}
          <Card className="glass border-primary/20 glow-primary">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="h-5 w-5 text-primary" />
                <span>Key Insights</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-primary mt-2"></div>
                <p className="text-sm">
                  <strong className="text-primary">ML-Hybrid</strong> outperforms traditional algorithms by{" "}
                  <strong>38%</strong> in average wait time
                </p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-secondary mt-2"></div>
                <p className="text-sm">
                  Energy consumption reduced by <strong className="text-success">40%</strong> compared to FCFS
                </p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent mt-2"></div>
                <p className="text-sm">
                  ML predictions achieved <strong className="text-accent">94% accuracy</strong> in process duration
                  estimation
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Compare;
