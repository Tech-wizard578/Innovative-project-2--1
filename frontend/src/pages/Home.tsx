import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Cpu,
  Brain,
  Zap,
  TrendingUp,
  Sparkles,
  ArrowRight,
  Shield,
  BarChart3,
  Activity,
} from "lucide-react";
import heroCpu from "@/assets/hero-cpu.jpg";
import aiNetwork from "@/assets/ai-network.jpg";
import Navbar from "@/components/Navbar";

const Home = () => {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Scheduling",
      description: "Machine learning algorithms predict optimal process allocation in real-time.",
    },
    {
      icon: Zap,
      title: "Energy Optimization",
      description: "Reduce power consumption by up to 40% with intelligent CPU state management.",
    },
    {
      icon: TrendingUp,
      title: "Performance Analytics",
      description: "Deep insights into wait times, throughput, and resource utilization.",
    },
    {
      icon: Shield,
      title: "Explainable AI",
      description: "Understand exactly why the scheduler makes each decision with XAI insights.",
    },
  ];

  const stats = [
    { value: "40%", label: "Energy Savings" },
    { value: "3x", label: "Faster Execution" },
    { value: "99.9%", label: "Accuracy" },
    { value: "Real-time", label: "Predictions" },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-24 pb-16 lg:pt-32 lg:pb-24">
        {/* Background Image with Overlay */}
        <div className="absolute inset-0 z-0">
          <img
            src={heroCpu}
            alt="CPU Background"
            className="w-full h-full object-cover opacity-20"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-background via-background/95 to-background" />
        </div>

        {/* Animated Background Elements */}
        <div className="absolute inset-0 z-0">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full"
              style={{
                width: Math.random() * 4 + 2,
                height: Math.random() * 4 + 2,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                background: i % 2 === 0 ? "hsl(var(--glow-primary))" : "hsl(var(--glow-secondary))",
              }}
              animate={{
                y: [0, -30, 0],
                opacity: [0.2, 0.8, 0.2],
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>

        <div className="container mx-auto px-4 lg:px-8 relative z-10">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-full glass border border-primary/20 glow-primary"
            >
              <Sparkles className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">AI-Powered CPU Scheduling Suite</span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-5xl lg:text-7xl font-bold leading-tight"
            >
              <span className="gradient-text">Intelligent CPU Scheduling</span>
              <br />
              <span className="text-foreground">That Learns & Optimizes</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-xl text-muted-foreground max-w-2xl mx-auto"
            >
              Harness the power of machine learning to predict, optimize, and explain CPU scheduling
              decisions in real-time. Reduce energy consumption while maximizing performance.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link to="/builder">
                <Button variant="hero" size="lg" className="group">
                  Run Scheduler
                  <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Link to="/dashboard">
                <Button variant="outline" size="lg">
                  <Activity className="mr-2 h-4 w-4" />
                  Live Dashboard
                </Button>
              </Link>
              <Link to="/compare">
                <Button variant="ghost" size="lg">
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Compare Algorithms
                </Button>
              </Link>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="grid grid-cols-2 lg:grid-cols-4 gap-6 pt-12"
            >
              {stats.map((stat, index) => (
                <div key={index} className="glass rounded-xl p-6 border border-border/50">
                  <div className="text-3xl font-bold gradient-text">{stat.value}</div>
                  <div className="text-sm text-muted-foreground mt-1">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 relative">
        <div className="absolute inset-0 z-0">
          <img
            src={aiNetwork}
            alt="AI Network"
            className="w-full h-full object-cover opacity-10"
          />
        </div>

        <div className="container mx-auto px-4 lg:px-8 relative z-10">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">
              <span className="gradient-text">Next-Generation</span> Scheduling
            </h2>
            <p className="text-xl text-muted-foreground">
              Powered by cutting-edge AI and designed for the future of computing
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="glass rounded-2xl p-8 border border-border/50 hover:border-primary/50 transition-all duration-300 group"
              >
                <div className="p-3 rounded-lg bg-gradient-to-br from-primary to-secondary w-fit mb-4 glow-primary group-hover:scale-110 transition-transform">
                  <feature.icon className="h-6 w-6 text-primary-foreground" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="container mx-auto px-4 lg:px-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="glass rounded-3xl p-12 lg:p-16 border border-primary/20 glow-primary text-center max-w-4xl mx-auto"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              Ready to optimize your <span className="gradient-text">CPU scheduling?</span>
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Start building intelligent process schedules with AI-powered predictions and
              real-time analytics.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/builder">
                <Button variant="glow" size="lg" className="group">
                  <Cpu className="mr-2 h-5 w-5" />
                  Launch Process Builder
                  <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
