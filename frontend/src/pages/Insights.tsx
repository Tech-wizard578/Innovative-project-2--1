import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, Lightbulb, Target } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import { ResponsiveContainer, RadialBarChart, RadialBar, Legend, PolarAngleAxis } from "recharts";

const Insights = () => {
  const confidenceData = [
    { name: "Confidence", value: 94, fill: "hsl(var(--primary))" },
  ];

  const featureImportance = [
    { feature: "Process Burst Time", weight: 0.35, color: "bg-primary" },
    { feature: "Arrival Time", weight: 0.25, color: "bg-secondary" },
    { feature: "Priority Level", weight: 0.20, color: "bg-accent" },
    { feature: "Process Type", weight: 0.15, color: "bg-success" },
    { feature: "CPU Affinity", weight: 0.05, color: "bg-muted" },
  ];

  const alternativeAlgorithms = [
    { name: "Priority-Based", score: 88, reason: "Better for real-time tasks" },
    { name: "SJF", score: 82, reason: "Minimal wait time for short processes" },
    { name: "Round Robin", score: 75, reason: "Fair time sharing" },
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
              <span className="gradient-text">Explainable AI Insights</span>
            </h1>
            <p className="text-muted-foreground">
              Understand the reasoning behind AI scheduling decisions
            </p>
          </div>

          {/* Main Decision Card */}
          <Card className="glass border-primary/20 glow-primary">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-3 rounded-lg bg-gradient-to-br from-primary to-secondary glow-primary">
                    <Brain className="h-6 w-6 text-primary-foreground" />
                  </div>
                  <div>
                    <CardTitle>AI Decision Summary</CardTitle>
                    <CardDescription>Current scheduling strategy explanation</CardDescription>
                  </div>
                </div>
                <Badge className="bg-primary/20 text-primary border-primary/50">
                  ML-Hybrid Active
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ duration: 1 }}
                className="space-y-2"
              >
                <p className="text-lg leading-relaxed">
                  The AI scheduler selected <strong className="text-primary">ML-Hybrid algorithm</strong> based
                  on the current workload characteristics. This decision optimizes for{" "}
                  <strong className="text-secondary">minimal wait time</strong> while maintaining{" "}
                  <strong className="text-accent">high throughput</strong> and{" "}
                  <strong className="text-success">energy efficiency</strong>.
                </p>
                <p className="text-muted-foreground">
                  The model detected a mixed workload pattern with 60% CPU-bound and 40% interactive processes,
                  making the hybrid approach most suitable for this scenario.
                </p>
              </motion.div>
            </CardContent>
          </Card>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Confidence Score */}
            <Card className="glass border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5 text-primary" />
                  <span>Decision Confidence</span>
                </CardTitle>
                <CardDescription>AI model confidence in current strategy</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <RadialBarChart
                    innerRadius="60%"
                    outerRadius="100%"
                    data={confidenceData}
                    startAngle={180}
                    endAngle={0}
                  >
                    <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                    <RadialBar
                      background
                      dataKey="value"
                      cornerRadius={10}
                      fill="hsl(var(--primary))"
                      className="glow-primary"
                    />
                    <text
                      x="50%"
                      y="50%"
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="fill-foreground text-5xl font-bold"
                    >
                      94%
                    </text>
                  </RadialBarChart>
                </ResponsiveContainer>
                <p className="text-center text-sm text-muted-foreground mt-4">
                  High confidence indicates strong model certainty in optimal algorithm selection
                </p>
              </CardContent>
            </Card>

            {/* Feature Importance */}
            <Card className="glass border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5 text-secondary" />
                  <span>Feature Importance</span>
                </CardTitle>
                <CardDescription>Factors influencing the AI decision (SHAP values)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {featureImportance.map((feature, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="space-y-2"
                  >
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{feature.feature}</span>
                      <span className="text-muted-foreground">{(feature.weight * 100).toFixed(0)}%</span>
                    </div>
                    <div className="relative h-3 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${feature.weight * 100}%` }}
                        transition={{ duration: 1, delay: index * 0.1 }}
                        className={`h-full ${feature.color} rounded-full`}
                      />
                    </div>
                  </motion.div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Alternative Recommendations */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Lightbulb className="h-5 w-5 text-accent" />
                <span>Alternative Algorithm Recommendations</span>
              </CardTitle>
              <CardDescription>Other viable options with trade-off analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alternativeAlgorithms.map((algo, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-4 glass-dark rounded-lg border border-border/50"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="font-semibold">{algo.name}</h4>
                        <Badge variant="outline">Score: {algo.score}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{algo.reason}</p>
                    </div>
                    <div className="ml-4">
                      <div className="w-16 h-16 rounded-full border-4 border-primary/20 flex items-center justify-center">
                        <span className="text-lg font-bold text-primary">{algo.score}</span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Reasoning Flowchart */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Decision Reasoning Flow</CardTitle>
              <CardDescription>Step-by-step breakdown of the AI decision process</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    step: 1,
                    title: "Workload Analysis",
                    description: "Analyzed 8 processes with mixed CPU-bound and interactive characteristics",
                  },
                  {
                    step: 2,
                    title: "Pattern Recognition",
                    description: "Identified bursty arrival pattern with high priority variance",
                  },
                  {
                    step: 3,
                    title: "Algorithm Scoring",
                    description: "Evaluated 6 algorithms based on predicted performance metrics",
                  },
                  {
                    step: 4,
                    title: "Optimization Selection",
                    description: "Selected ML-Hybrid for optimal balance of all key metrics",
                  },
                ].map((step, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.15 }}
                    className="flex items-start space-x-4 p-4 glass-dark rounded-lg border border-border/50"
                  >
                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center font-bold text-primary-foreground glow-primary">
                      {step.step}
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold mb-1">{step.title}</h4>
                      <p className="text-sm text-muted-foreground">{step.description}</p>
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

export default Insights;
