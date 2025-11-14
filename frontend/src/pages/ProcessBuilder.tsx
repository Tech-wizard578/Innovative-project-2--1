import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Cpu, MemoryStick, Timer, Zap, Trash2 } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";

// --- NEW IMPORTS ---
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useProcesses, Process } from "@/contexts/ProcessContext";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
// --- END NEW IMPORTS ---

// 1. Define the form's validation schema with Zod
const formSchema = z.object({
  arrival_time: z.coerce.number().min(0, "Arrival time must be >= 0"),
  burst_time: z.coerce.number().min(1, "Burst time must be > 0"),
  priority: z.coerce.number().min(1, "Priority must be at least 1").max(10, "Priority can be at most 10"),
  process_type: z.coerce.number().min(0).max(3),
});

const ProcessBuilder = () => {
  // 2. Get process state and functions from our global context
  const { processes, addProcess, clearProcesses } = useProcesses();
  const navigate = useNavigate();

  // 3. Set up the form using react-hook-form and Zod
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      arrival_time: 0,
      burst_time: 10,
      priority: 5,
      process_type: 0,
    },
  });

  // 4. Handle form submission
  const onSubmit = (values: z.infer<typeof formSchema>) => {
    // Add the valid data to our global process list
    addProcess(values);
    toast.success(`Process P${processes.length + 1} added to queue!`);
    form.reset();
  };

  // 5. Handle navigation to the scheduler
  const onRunSimulation = () => {
    if (processes.length === 0) {
      toast.error("Please add at least one process to the queue.");
      return;
    }
    navigate("/scheduler");
  };

  // Helper to get process type name
  const getProcessTypeName = (type: number) => {
    switch (type) {
      case 0: return "CPU-Bound";
      case 1: return "I/O-Bound";
      case 2: return "Mixed";
      case 3: return "Interactive";
      default: return "Unknown";
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="container mx-auto px-4 lg:px-8 pt-24 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-8"
        >
          {/* Form Card */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>AI Process Builder</CardTitle>
              <CardDescription>Define process characteristics for the AI scheduler</CardDescription>
            </CardHeader>
            <CardContent>
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="arrival_time"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Arrival Time (ms)</FormLabel>
                          <FormControl>
                            <Input type="number" {...field} onChange={(e) => field.onChange(parseInt(e.target.value) || 0)} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="burst_time"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Est. Burst Time (ms)</FormLabel>
                          <FormControl>
                            <Input type="number" {...field} onChange={(e) => field.onChange(parseInt(e.target.value) || 0)} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="priority"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Priority (1-10)</FormLabel>
                          <FormControl>
                            <Input type="number" {...field} onChange={(e) => field.onChange(parseInt(e.target.value) || 0)} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="process_type"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Process Type</FormLabel>
                          <Select onValueChange={field.onChange} value={String(field.value)}>
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select type..." />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="0">CPU-Bound</SelectItem>
                              <SelectItem value="1">I/O-Bound</SelectItem>
                              <SelectItem value="2">Mixed</SelectItem>
                              <SelectItem value="3">Interactive</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                  <div className="text-center pt-2">
                    <Button type="submit" className="w-full" variant="default">
                      <Zap className="h-4 w-4 mr-2" />
                      Add Process to Queue
                    </Button>
                  </div>
                </form>
              </Form>
            </CardContent>
          </Card>

          {/* Queue Card */}
          <Card className="glass border-border/50">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Process Queue</CardTitle>
                <CardDescription>Processes ready for simulation</CardDescription>
              </div>
              {processes.length > 0 && (
                <Button variant="ghost" size="icon" onClick={clearProcesses}>
                  <Trash2 className="h-4 w-4 text-muted-foreground" />
                </Button>
              )}
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3 h-48 overflow-y-auto pr-2">
                {processes.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-muted-foreground">
                    Add processes using the form to see them here.
                  </div>
                ) : (
                  processes.map((proc: Process) => (
                    <div key={proc.pid} className="flex items-center justify-between p-3 glass-dark rounded-lg border border-border/50">
                      <div className="flex items-center space-x-3">
                        <Badge variant="outline">P{proc.pid}</Badge>
                        <div>
                          <div className="font-semibold">{getProcessTypeName(proc.process_type)}</div>
                          <div className="text-xs text-muted-foreground">Priority: {proc.priority} | Arrival: {proc.arrival_time}ms</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{proc.burst_time}ms</div>
                        <div className="text-xs text-muted-foreground">Burst Time</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <Button className="w-full" variant="outline" onClick={onRunSimulation}>
                Run Simulation
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default ProcessBuilder;
