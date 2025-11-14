// src/App.tsx
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ProcessProvider } from "@/contexts/ProcessContext";

// --- Page Imports ---
import Home from "./pages/Home";
import ProcessBuilder from "./pages/ProcessBuilder";
import Scheduler from "./pages/Scheduler";
import NotFound from "./pages/NotFound";

// --- ADD IMPORTS FOR YOUR OTHER PAGES HERE ---
import Compare from "./pages/Compare"; 
import XAIInsights from "./pages/Insights";
import EnergyDashboard from "./pages/Energy";
// ... add any other page imports you need ...

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <TooltipProvider>
        <ProcessProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              {/* --- Core Routes --- */}
              <Route path="/" element={<Home />} />
              <Route path="/builder" element={<ProcessBuilder />} />
              <Route path="/scheduler" element={<Scheduler />} />

              {/* --- ADDED ROUTES TO FIX 404 ERRORS --- */}
              <Route path="/compare" element={<Compare />} />
              <Route path="/xai-insights" element={<XAIInsights />} />
              <Route path="/energy-dashboard" element={<EnergyDashboard />} />
              
              {/* ... add any other routes here ... */}

              {/* --- Fallback 404 Route (Must be last) --- */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </ProcessProvider>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;