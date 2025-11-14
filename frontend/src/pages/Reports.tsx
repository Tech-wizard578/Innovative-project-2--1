import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, FileJson, FileText } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";

const Reports = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="container mx-auto px-4 lg:px-8 pt-24 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto space-y-8"
        >
          <div>
            <h1 className="text-4xl font-bold mb-2">
              <span className="gradient-text">Export & Reports</span>
            </h1>
            <p className="text-muted-foreground">Generate comprehensive scheduling reports</p>
          </div>

          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Export Options</CardTitle>
              <CardDescription>Download scheduling data and analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <Button variant="outline" size="lg" className="h-auto py-6">
                  <div className="flex flex-col items-center space-y-2">
                    <FileJson className="h-8 w-8 text-primary" />
                    <span>Export JSON</span>
                    <span className="text-xs text-muted-foreground">Raw data format</span>
                  </div>
                </Button>
                <Button variant="outline" size="lg" className="h-auto py-6">
                  <div className="flex flex-col items-center space-y-2">
                    <FileText className="h-8 w-8 text-secondary" />
                    <span>Generate PDF</span>
                    <span className="text-xs text-muted-foreground">Formatted report</span>
                  </div>
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle>Report Preview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                Report generator coming soon...
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Reports;
