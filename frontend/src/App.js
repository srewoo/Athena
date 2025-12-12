import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { lazy, Suspense } from "react";
import PromptOptimizer from "./pages/PromptOptimizer";
import { Toaster } from "./components/ui/toaster";
import { ThemeProvider } from "./components/theme-provider";
import ErrorBoundary from "./components/ErrorBoundary";
import "./App.css";

// Lazy load pages for better performance
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Playground = lazy(() => import("./pages/Playground"));
const ContradictionDetector = lazy(() => import("./pages/ContradictionDetector"));
const DelimiterAnalyzer = lazy(() => import("./pages/DelimiterAnalyzer"));
const MetapromptGenerator = lazy(() => import("./pages/MetapromptGenerator"));
const History = lazy(() => import("./pages/History"));
const Compare = lazy(() => import("./pages/Compare"));
const ABTesting = lazy(() => import("./pages/ABTesting"));
const Help = lazy(() => import("./pages/Help"));
const EvaluationDetail = lazy(() => import("./pages/EvaluationDetail"));

// API base URL - adjust this if your backend runs on a different port
export const BASE_URL = process.env.REACT_APP_BASE_URL || "http://localhost:8000";
export const API = process.env.REACT_APP_API_URL || `${BASE_URL}/api`;

// Loading fallback component
const PageLoader = () => (
  <div className="min-h-screen flex items-center justify-center">
    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
  </div>
);

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="athena-theme">
      <ErrorBoundary>
        <Router>
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route path="/" element={<PromptOptimizer />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/playground" element={<Playground />} />
                <Route path="/contradiction-detector" element={<ContradictionDetector />} />
                <Route path="/delimiter-analyzer" element={<DelimiterAnalyzer />} />
                <Route path="/metaprompt-generator" element={<MetapromptGenerator />} />
                <Route path="/history" element={<History />} />
                <Route path="/compare" element={<Compare />} />
                <Route path="/ab-testing" element={<ABTesting />} />
                <Route path="/help" element={<Help />} />
                <Route path="/evaluation/:id" element={<EvaluationDetail />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
            <Toaster />
          </div>
        </Router>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App;

