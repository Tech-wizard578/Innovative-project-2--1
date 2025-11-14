import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Leaf, Zap, TrendingDown, Battery } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Area,
  AreaChart,
} from "recharts";

const Energy = () => {
  const powerData = [
    { time: "0s", power: 45, baseline: 60 },
    { time: "5s", power: 38, baseline: 58 },
    { time: "10s", power: 42, baseline: 62 },
    { time: "15s", power: 28, baseline: 59 },
    { time: "20s", power: 35, baseline: 61 },
    { time: "25s", power: 30, baseline: 60 },
  ];

  const cpuStates = [
    { state: "Active", duration: "45%", power: "High", color: "bg-destructive" },
    { state: "Turbo", duration: "15%", power: "Very High", color: "bg-primary" },
    { state: "Power Save", duration: "40%", power: "Low", color: "bg-success" },
  ];

  const metrics = [
    {
      label: "Energy Saved",
      value: "42%",
      icon: TrendingDown,
      description: "vs baseline FCFS",
      color: "text-success",
    },
    {
      label: "CO₂ Reduced",
      value: "18kg",
      icon: Leaf,
      description: "per 1000 hours",
      color: "text-success",
    },
    {
      label: "Avg Power",
      value: "36W",
      icon: Zap,
      description: "current workload",
      color: "text-primary",
    },
    {
      label: "Cost Savings",
      value: "$42",
      icon: Battery,
      description: "per month",
      color: "text-accent",
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
              <span className="gradient-text">Energy & Sustainability</span>
            </h1>
            <p className="text-muted-foreground">
              Monitor power consumption and environmental impact
            </p>
          </div>

          {/* Metrics Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
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
                      <metric.icon className={`h-6 w-6 ${metric.color}`} />
                    </div>
                    <div className="text-3xl font-bold mb-1">{metric.value}</div>
                    <div className="text-sm text-muted-foreground">{metric.label}</div>
                    <div className="text-xs text-muted-foreground mt-1">{metric.description}</div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Power Consumption Timeline */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Real-Time Power Consumption</CardTitle>
              <CardDescription>
                Actual vs baseline power usage over time (watts)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={powerData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" tick={{ fill: "hsl(var(--foreground))" }} />
                  <YAxis tick={{ fill: "hsl(var(--muted-foreground))" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="baseline"
                    stroke="hsl(var(--muted))"
                    fill="hsl(var(--muted))"
                    fillOpacity={0.3}
                    name="Baseline (FCFS)"
                  />
                  <Area
                    type="monotone"
                    dataKey="power"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.5}
                    name="ML-Hybrid"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* CPU Power States */}
            <Card className="glass border-border/50">
              <CardHeader>
                <CardTitle>CPU Power State Distribution</CardTitle>
                <CardDescription>Time spent in each power state</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {cpuStates.map((state, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${state.color}`}></div>
                        <span className="font-medium">{state.state}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">{state.power}</Badge>
                        <span className="text-sm text-muted-foreground">{state.duration}</span>
                      </div>
                    </div>
                    <div className="relative h-3 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: state.duration }}
                        transition={{ duration: 1, delay: index * 0.1 }}
                        className={`h-full ${state.color} rounded-full`}
                      />
                    </div>
                  </motion.div>
                ))}
              </CardContent>
            </Card>

            {/* Environmental Impact */}
            <Card className="glass border-success/20 glow-primary">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Leaf className="h-5 w-5 text-success" />
                  <CardTitle>Environmental Impact Score</CardTitle>
                </div>
                <CardDescription>Sustainability metrics and carbon footprint</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="text-center">
                  <div className="text-6xl font-bold gradient-text mb-2">A+</div>
                  <p className="text-sm text-muted-foreground">
                    Excellent energy efficiency rating
                  </p>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between p-3 glass-dark rounded-lg">
                    <span className="text-sm">CO₂ per hour</span>
                    <span className="font-semibold text-success">0.018kg</span>
                  </div>
                  <div className="flex justify-between p-3 glass-dark rounded-lg">
                    <span className="text-sm">Energy efficiency</span>
                    <span className="font-semibold text-success">94%</span>
                  </div>
                  <div className="flex justify-between p-3 glass-dark rounded-lg">
                    <span className="text-sm">Equivalent trees</span>
                    <span className="font-semibold text-success">3.2 saved/month</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Energy Optimization Recommendations</CardTitle>
              <CardDescription>AI-suggested improvements for better efficiency</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  {
                    title: "Increase Power Save Duration",
                    impact: "+8% energy savings",
                    description: "Extend power save state during idle periods",
                  },
                  {
                    title: "Optimize Process Clustering",
                    impact: "+5% efficiency",
                    description: "Group similar processes to reduce context switches",
                  },
                  {
                    title: "Dynamic Frequency Scaling",
                    impact: "+12% reduction",
                    description: "Implement adaptive CPU frequency based on load",
                  },
                ].map((rec, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-start space-x-4 p-4 glass-dark rounded-lg border border-border/50 hover:border-primary/50 transition-colors"
                  >
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-success/20 flex items-center justify-center">
                      <Zap className="h-4 w-4 text-success" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-semibold">{rec.title}</h4>
                        <Badge className="bg-success/20 text-success border-success/50">
                          {rec.impact}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{rec.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Energy;
