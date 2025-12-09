import React, { useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Save, Download, RefreshCw, Settings, Eye, EyeOff, MessageSquare, Send, X, FolderOpen, Plus, Trash2, Pencil, Play, Square, CheckCircle, XCircle, AlertCircle, BarChart3, ArrowUpDown, RotateCcw, FileText, Maximize2, TrendingUp, TrendingDown, Minus } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../components/ui/dialog";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { useToast } from "../hooks/use-toast";
import { API } from "../App";
import { ThemeToggle } from "../components/theme-toggle";

const PromptOptimizer = () => {
  const { toast } = useToast();
  const [expandedSections, setExpandedSections] = useState({
    requirements: true,
    optimization: false,
    evalPrompt: false,
    dataset: false,
    testRun: false
  });

  // Section 1: Requirements & Initial Prompt
  const [projectName, setProjectName] = useState("");
  const [useCase, setUseCase] = useState("");
  const [keyRequirements, setKeyRequirements] = useState("");
  const [targetProvider, setTargetProvider] = useState("openai");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [projectId, setProjectId] = useState(null);
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  // Project Management
  const [savedProjects, setSavedProjects] = useState([]);
  const [projectSelectorOpen, setProjectSelectorOpen] = useState(false);
  const [isLoadingProjects, setIsLoadingProjects] = useState(false);
  const [isLoadingProject, setIsLoadingProject] = useState(false);

  // Section 2: Optimization
  const [analysisResults, setAnalysisResults] = useState(null);
  const [currentVersion, setCurrentVersion] = useState(null);
  const [versionHistory, setVersionHistory] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);
  const [showPromptFeedback, setShowPromptFeedback] = useState(false);
  const [promptFeedback, setPromptFeedback] = useState("");
  const [isRefiningPrompt, setIsRefiningPrompt] = useState(false);
  const [promptChanges, setPromptChanges] = useState([]);

  // Section 3: Eval Prompt
  const [evalPrompt, setEvalPrompt] = useState("");
  const [evalRationale, setEvalRationale] = useState("");
  const [isGeneratingEval, setIsGeneratingEval] = useState(false);
  const [showEvalFeedback, setShowEvalFeedback] = useState(false);
  const [evalFeedback, setEvalFeedback] = useState("");
  const [isRefiningEval, setIsRefiningEval] = useState(false);
  const [evalChanges, setEvalChanges] = useState([]);

  // Section 4: Dataset
  const [dataset, setDataset] = useState(null);
  const [isCurrentSessionDataset, setIsCurrentSessionDataset] = useState(false); // Track if dataset is from current session
  const [sampleCount, setSampleCount] = useState(100);
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false);
  const [datasetProgress, setDatasetProgress] = useState({ progress: 0, batch: 0, total_batches: 0, status: '' });
  const [serverStatus, setServerStatus] = useState("checking"); // checking, online, offline

  // Section 5: Test Run
  const [testRunId, setTestRunId] = useState(null);
  const [testRunStatus, setTestRunStatus] = useState(null); // { status, progress, completed_items, total_items, ... }
  const [testRunResults, setTestRunResults] = useState(null);
  const [isStartingTestRun, setIsStartingTestRun] = useState(false);
  const [testRunHistory, setTestRunHistory] = useState([]);
  const [selectedTestRunVersion, setSelectedTestRunVersion] = useState(null);
  const [testRunPassThreshold, setTestRunPassThreshold] = useState(3.5);

  // Comparison mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareRunA, setCompareRunA] = useState(null);
  const [compareRunB, setCompareRunB] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  // Single test mode
  const [singleTestOpen, setSingleTestOpen] = useState(false);
  const [singleTestInput, setSingleTestInput] = useState("");
  const [singleTestResult, setSingleTestResult] = useState(null);
  const [isRunningSingleTest, setIsRunningSingleTest] = useState(false);

  // Detail view modal
  const [detailViewOpen, setDetailViewOpen] = useState(false);
  const [detailViewItem, setDetailViewItem] = useState(null);

  // Re-run failed
  const [isRerunningFailed, setIsRerunningFailed] = useState(false);

  // Regenerate dataset dialog
  const [regenerateDialogOpen, setRegenerateDialogOpen] = useState(false);
  const [regenerateSampleCount, setRegenerateSampleCount] = useState(100);

  // Settings Modal
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [llmProvider, setLlmProvider] = useState("openai");
  const [llmModel, setLlmModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  // Model options per provider (2025 latest releases)
  const modelOptions = {
    openai: [
      "gpt-5",  // Latest flagship (2025)
      "gpt-4.1",  // April 2025, ~1M context
      "gpt-4o",
      "gpt-4o-mini",
      { label: "--- Reasoning Models ---", disabled: true },
      "o1",
      "o1-preview",
      "o1-mini",
      "o3",
      "o3-mini"
    ],
    claude: [
      "claude-sonnet-4-5-20250929",  // Best coding model (Sep 2025)
      "claude-haiku-4-5-20251001",  // Fast, low-latency (Oct 2025)
      "claude-3-7-sonnet-20250219",
      "claude-3-5-sonnet-20241022"
    ],
    gemini: [
      "gemini-2.5-pro",  // Hybrid-reasoning (2025)
      "gemini-2.5-flash",  // Fast variant (2025)
      "gemini-2.0-flash-exp"
    ]
  };

  // State for thinking mode
  const [useThinkingMode, setUseThinkingMode] = useState(false);

  // Load existing settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await fetch(`${API}/settings`);
        if (response.ok) {
          const data = await response.json();
          if (data) {
            setLlmProvider(data.llm_provider || "openai");
            setLlmModel(data.model_name || "");
            setApiKey(data.api_key || "");
            setSettingsLoaded(true);
          }
        }
      } catch (error) {
        console.error("Failed to load settings:", error);
      }
    };
    loadSettings();
  }, []);

  // Save settings handler
  const handleSaveSettings = async () => {
    if (!apiKey.trim()) {
      toast({
        title: "API Key Required",
        description: "Please enter your API key",
        variant: "destructive"
      });
      return;
    }

    setIsSavingSettings(true);
    try {
      const response = await fetch(`${API}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm_provider: llmProvider,
          api_key: apiKey,
          model_name: llmModel || modelOptions[llmProvider][0]
        })
      });

      if (response.ok) {
        toast({
          title: "Settings Saved",
          description: `Using ${llmProvider.toUpperCase()} for AI operations`
        });
        setSettingsOpen(false);
        setSettingsLoaded(true);
      } else {
        throw new Error("Failed to save settings");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save settings",
        variant: "destructive"
      });
    } finally {
      setIsSavingSettings(false);
    }
  };

  // Check server status on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const response = await fetch(`${API}/projects`, { method: "GET" });
        setServerStatus(response.ok ? "online" : "offline");
      } catch (error) {
        setServerStatus("offline");
      }
    };
    checkServer();
  }, []);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Load saved projects from database
  const loadSavedProjects = async () => {
    setIsLoadingProjects(true);
    try {
      const response = await fetch(`${API}/projects?limit=50`);
      if (response.ok) {
        const projects = await response.json();
        setSavedProjects(projects);
      }
    } catch (error) {
      console.error("Failed to load projects:", error);
    } finally {
      setIsLoadingProjects(false);
    }
  };

  // Load a specific project
  const loadProject = async (id) => {
    setIsLoadingProject(true);
    try {
      const response = await fetch(`${API}/projects/${id}`);
      if (!response.ok) throw new Error("Failed to load project");

      const project = await response.json();

      // Set project data
      setProjectId(project.id);
      setProjectName(project.name);
      setUseCase(project.requirements.use_case);
      setKeyRequirements(project.requirements.key_requirements.join("\n"));
      setTargetProvider(project.requirements.target_provider);

      // Set versions
      if (project.system_prompt_versions && project.system_prompt_versions.length > 0) {
        setVersionHistory(project.system_prompt_versions);
        const latestVersion = project.system_prompt_versions[project.system_prompt_versions.length - 1];
        setCurrentVersion(latestVersion);
        setInitialPrompt(project.system_prompt_versions[0].prompt_text);
      }

      // Set eval prompt if exists
      if (project.eval_prompt) {
        setEvalPrompt(project.eval_prompt.prompt_text);
        setEvalRationale(project.eval_prompt.rationale);
      }

      // Set dataset if exists (but mark as not from current session)
      if (project.dataset) {
        setDataset(project.dataset);
        setIsCurrentSessionDataset(false); // Loaded from saved project, not current session
      }

      // Expand optimization section
      setExpandedSections({
        requirements: false,
        optimization: true,
        evalPrompt: false,
        dataset: false
      });

      setProjectSelectorOpen(false);

      toast({
        title: "Project Loaded",
        description: `Loaded "${project.name}"`
      });

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsLoadingProject(false);
    }
  };

  // Reset to create new project
  const resetToNewProject = () => {
    setProjectId(null);
    setProjectName("");
    setUseCase("");
    setKeyRequirements("");
    setTargetProvider("openai");
    setInitialPrompt("");
    setAnalysisResults(null);
    setCurrentVersion(null);
    setVersionHistory([]);
    setEvalPrompt("");
    setEvalRationale("");
    setEvalChanges([]);
    setDataset(null);
    setIsCurrentSessionDataset(false);
    setPromptChanges([]);
    setExpandedSections({
      requirements: true,
      optimization: false,
      evalPrompt: false,
      dataset: false
    });
    setProjectSelectorOpen(false);
  };

  const handleEditProject = async (project, event) => {
    // Stop event propagation to prevent loading the project when clicking edit
    event.stopPropagation();

    // Load the project into the form fields for editing
    setProjectId(project.id);
    setProjectName(project.name);
    setUseCase(project.requirements?.use_case || "");
    setKeyRequirements(project.requirements?.key_requirements?.join(", ") || "");
    setInitialPrompt(project.system_prompt_versions?.[0]?.prompt_text || "");
    setTargetProvider(project.requirements?.target_provider || "openai");

    // Reset other state to start fresh
    setCurrentVersion(null);
    setVersionHistory([]);
    setAnalysisResults(null);
    setEvalPrompt("");
    setDataset(null);

    // Close the project selector modal
    setProjectSelectorOpen(false);

    // Expand the requirements section
    setExpandedSections(prev => ({
      ...prev,
      requirements: true,
      optimization: false,
      evalPrompt: false,
      dataset: false
    }));

    toast({
      title: "Project Loaded for Editing",
      description: `You can now edit "${project.name}" and save your changes`
    });
  };

  const handleDeleteProject = async (projectIdToDelete, projectNameToDelete, event) => {
    // Stop event propagation to prevent loading the project when clicking delete
    event.stopPropagation();

    // Confirm deletion
    const confirmed = window.confirm(
      `Are you sure you want to delete "${projectNameToDelete}"?\n\nThis action cannot be undone and will delete all versions and associated data.`
    );

    if (!confirmed) return;

    try {
      const response = await fetch(`${API}/projects/${projectIdToDelete}`, {
        method: "DELETE"
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete project");
      }

      // Remove from local state
      setSavedProjects(prev => prev.filter(p => p.id !== projectIdToDelete));

      // If this was the current project, reset to new project
      if (projectId === projectIdToDelete) {
        resetToNewProject();
      }

      toast({
        title: "Project Deleted",
        description: `"${projectNameToDelete}" has been deleted successfully`
      });

    } catch (error) {
      toast({
        title: "Delete Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  const handleCreateProject = async () => {
    try {
      // Parse requirements (comma or newline separated)
      const reqList = keyRequirements
        .split(/[,\n]/)
        .map(r => r.trim())
        .filter(r => r.length > 0);

      if (!projectName || !useCase || reqList.length === 0 || !initialPrompt) {
        toast({
          title: "Missing Fields",
          description: "Please fill in all required fields",
          variant: "destructive"
        });
        return;
      }

      setIsCreatingProject(true);

      toast({
        title: "Creating Project...",
        description: "Please wait while we set up your project and analyze your prompt. This may take 20-30 seconds.",
      });

      const response = await fetch(`${API}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: projectName,
          use_case: useCase,
          key_requirements: reqList,
          target_provider: targetProvider,
          initial_prompt: initialPrompt
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to create project. Please ensure the backend server is running.");
      }

      const project = await response.json();
      setProjectId(project.id);
      setVersionHistory(project.system_prompt_versions || []);
      setCurrentVersion(project.system_prompt_versions?.[0] || null);

      toast({
        title: "Project Created",
        description: "Your project has been created successfully. Now analyzing your prompt..."
      });

      // Auto-expand optimization section
      setExpandedSections(prev => ({
        ...prev,
        requirements: false,
        optimization: true
      }));

      // Auto-analyze (pass project ID directly since state update is async)
      await handleAnalyze(initialPrompt, project.id);

    } catch (error) {
      console.error("Project creation error:", error);
      toast({
        title: "Connection Error",
        description: error.message.includes("fetch")
          ? "Cannot connect to the backend server. Please ensure the server is running on port 8000 and restart it if needed."
          : error.message,
        variant: "destructive"
      });
    } finally {
      setIsCreatingProject(false);
    }
  };

  const handleAnalyze = async (promptText = initialPrompt, idOverride = null, versionOverride = null) => {
    const projectIdToUse = idOverride || projectId;
    if (!projectIdToUse && !promptText) return;

    setIsAnalyzing(true);
    try {
      const response = await fetch(`${API}/projects/${projectIdToUse}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_text: promptText })
      });

      if (!response.ok) throw new Error("Failed to analyze prompt");

      const results = await response.json();
      setAnalysisResults(results);

      // Update the current version's evaluation in version history
      // Use versionOverride if provided, otherwise use currentVersion from state
      const versionToUpdate = versionOverride || currentVersion;
      if (versionToUpdate) {
        const evaluation = {
          requirements_alignment: results.requirements_alignment_score || 0,
          best_practices_score: results.best_practices_score || 0,
          suggestions: results.suggestions || [],
          requirements_gaps: results.requirements_gaps || []
        };

        // Update version history with the evaluation
        setVersionHistory(prev => prev.map(v =>
          v.version === versionToUpdate.version
            ? { ...v, evaluation }
            : v
        ));

        // Also update current version
        setCurrentVersion(prev => ({ ...prev, evaluation }));
      }

      toast({
        title: "Analysis Complete",
        description: `Overall score: ${results.overall_score.toFixed(1)}/100`
      });

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRewrite = async () => {
    if (!projectId || !currentVersion) return;

    // Check if we have analysis results with suggestions to incorporate
    if (!analysisResults || !analysisResults.suggestions || analysisResults.suggestions.length === 0) {
      toast({
        title: "No Suggestions Available",
        description: "Please run Re-Analyze first to get suggestions to incorporate",
        variant: "destructive"
      });
      return;
    }

    setIsRewriting(true);
    try {
      // Build focus areas from the analysis suggestions
      const focusAreas = analysisResults.suggestions.slice(0, 5).map(sug => {
        // Extract the suggestion text (remove priority prefix if present)
        if (typeof sug === 'string') {
          return sug;
        }
        return sug.suggestion || sug.text || String(sug);
      });

      // Include requirements gaps as additional focus areas
      if (analysisResults.requirements_gaps && analysisResults.requirements_gaps.length > 0) {
        analysisResults.requirements_gaps.slice(0, 3).forEach(gap => {
          focusAreas.push(`Address requirement gap: ${gap}`);
        });
      }

      const response = await fetch(`${API}/rewrite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: currentVersion.prompt_text,
          focus_areas: focusAreas,
          use_case: useCase,  // Pass the original use case
          key_requirements: keyRequirements  // Pass the key requirements
        })
      });

      if (!response.ok) throw new Error("Failed to rewrite prompt");

      const result = await response.json();

      // Add new version with the changes made
      // Backend returns "rewritten_prompt", handle both for compatibility
      const newPrompt = result.rewritten_prompt || result.improved_prompt;
      if (!newPrompt) {
        throw new Error("No improved prompt returned from server");
      }
      const changesDescription = result.changes_made ? result.changes_made.join("; ") : "AI improvements applied";
      await addVersion(newPrompt, changesDescription);

      toast({
        title: "Suggestions Incorporated",
        description: `Applied ${focusAreas.length} improvements. New version is being analyzed...`
      });

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRewriting(false);
    }
  };

  const handleRefinePromptWithFeedback = async () => {
    if (!projectId || !currentVersion || !promptFeedback.trim()) {
      toast({
        title: "Feedback Required",
        description: "Please provide feedback for refinement",
        variant: "destructive"
      });
      return;
    }

    if (!settingsLoaded || !apiKey) {
      toast({
        title: "Settings Required",
        description: "Please configure your LLM settings first (click the gear icon)",
        variant: "destructive"
      });
      return;
    }

    setIsRefiningPrompt(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/rewrite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_prompt: currentVersion.prompt_text,
          focus_areas: [promptFeedback]
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to refine prompt");
      }

      const result = await response.json();

      if (result.improved_prompt) {
        // Add new version
        await addVersion(result.improved_prompt, promptFeedback);
        setPromptChanges(result.changes_made || []);
        setPromptFeedback("");
        setShowPromptFeedback(false);

        toast({
          title: "Prompt Refined",
          description: `Applied ${result.changes_made?.length || 0} changes based on your feedback`
        });
      } else {
        throw new Error("No refined prompt returned");
      }
    } catch (error) {
      toast({
        title: "Refinement Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRefiningPrompt(false);
    }
  };

  const addVersion = async (promptText, feedback = "") => {
    try {
      const response = await fetch(`${API}/projects/${projectId}/versions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: promptText,
          user_feedback: feedback,
          is_final: false
        })
      });

      if (!response.ok) throw new Error("Failed to add version");

      const newVersion = await response.json();
      setVersionHistory(prev => [...prev, newVersion]);
      setCurrentVersion(newVersion);

      // Auto-analyze new version - pass the version object to avoid stale state
      handleAnalyze(promptText, null, newVersion);

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  const handleGenerateEvalPrompt = async () => {
    if (!projectId) return;

    setIsGeneratingEval(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });

      if (!response.ok) throw new Error("Failed to generate eval prompt");

      const result = await response.json();
      setEvalPrompt(result.eval_prompt);
      setEvalRationale(result.rationale);

      toast({
        title: "Eval Prompt Generated",
        description: "Your evaluation prompt is ready"
      });

      // Keep evalPrompt section expanded to show the generated prompt
      setExpandedSections(prev => ({
        ...prev,
        evalPrompt: true
      }));

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsGeneratingEval(false);
    }
  };

  const handleRefineEvalPrompt = async () => {
    if (!projectId || !evalFeedback.trim()) {
      toast({
        title: "Feedback Required",
        description: "Please provide feedback for refinement",
        variant: "destructive"
      });
      return;
    }

    if (!settingsLoaded || !apiKey) {
      toast({
        title: "Settings Required",
        description: "Please configure your LLM settings first (click the gear icon)",
        variant: "destructive"
      });
      return;
    }

    setIsRefiningEval(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_eval_prompt: evalPrompt,
          user_feedback: evalFeedback
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to refine eval prompt");
      }

      const result = await response.json();

      if (result.refined_prompt) {
        setEvalPrompt(result.refined_prompt);
        setEvalRationale(result.rationale || "Refined based on user feedback");
        setEvalChanges(result.changes_made || []);
        setEvalFeedback("");
        setShowEvalFeedback(false);

        toast({
          title: "Eval Prompt Refined",
          description: `Applied ${result.changes_made?.length || 0} changes based on your feedback`
        });
      } else {
        throw new Error("No refined prompt returned");
      }
    } catch (error) {
      toast({
        title: "Refinement Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRefiningEval(false);
    }
  };

  const handleGenerateDataset = async (overrideSampleCount = null) => {
    if (!projectId) return;

    // Use override if provided, otherwise use state
    const countToGenerate = overrideSampleCount ?? sampleCount;

    setIsGeneratingDataset(true);
    setDatasetProgress({ progress: 0, batch: 0, total_batches: 0, status: 'starting' });

    try {
      // Use streaming endpoint for large datasets with heartbeat
      const response = await fetch(`${API}/projects/${projectId}/dataset/generate-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_count: countToGenerate
        })
      });

      if (!response.ok) {
        // Fall back to regular endpoint if streaming fails
        const fallbackResponse = await fetch(`${API}/projects/${projectId}/dataset/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_count: countToGenerate
          })
        });

        if (!fallbackResponse.ok) throw new Error("Failed to generate dataset");

        const result = await fallbackResponse.json();
        setDataset(result);
        setIsCurrentSessionDataset(true); // Mark as current session dataset
        setDatasetProgress({ progress: 100, batch: 0, total_batches: 0, status: 'completed' });

        toast({
          title: "Dataset Generated",
          description: `${result.sample_count} test cases created`
        });
        return;
      }

      // Process SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              // Handle different message types
              if (data.type === 'heartbeat') {
                // Heartbeat received - connection is alive
                console.log('Heartbeat received:', data.timestamp);
              } else if (data.status === 'completed') {
                // Generation complete
                setDataset({
                  dataset_id: data.dataset_id,
                  csv_content: data.csv_content,
                  sample_count: data.sample_count,
                  preview: data.preview
                });
                setIsCurrentSessionDataset(true); // Mark as current session dataset
                setDatasetProgress({ progress: 100, batch: 0, total_batches: 0, status: 'completed' });

                toast({
                  title: "Dataset Generated",
                  description: `${data.sample_count} test cases created`
                });
              } else if (data.status === 'error') {
                throw new Error(data.message || 'Dataset generation failed');
              } else if (data.status === 'generating') {
                // Progress update
                setDatasetProgress({
                  progress: data.progress || 0,
                  batch: data.batch || 0,
                  total_batches: data.total_batches || 0,
                  status: 'generating'
                });
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
      setDatasetProgress({ progress: 0, batch: 0, total_batches: 0, status: 'error' });
    } finally {
      setIsGeneratingDataset(false);
    }
  };

  const handleDeleteVersion = async (versionNumber) => {
    if (!projectId) return;

    // Don't allow deleting the current version or if it's the only version
    if (versionHistory.length <= 1) {
      toast({
        title: "Cannot Delete",
        description: "Cannot delete the only version",
        variant: "destructive"
      });
      return;
    }

    if (currentVersion?.version === versionNumber) {
      toast({
        title: "Cannot Delete",
        description: "Cannot delete the currently active version. Switch to another version first.",
        variant: "destructive"
      });
      return;
    }

    try {
      const response = await fetch(`${API}/projects/${projectId}/versions/${versionNumber}`, {
        method: "DELETE"
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete version");
      }

      // Remove version from local state
      setVersionHistory(prev => prev.filter(v => v.version !== versionNumber));

      toast({
        title: "Version Deleted",
        description: `Version ${versionNumber} has been deleted`
      });

    } catch (error) {
      toast({
        title: "Delete Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  const handleDownloadDataset = async () => {
    if (!projectId) return;

    try {
      const response = await fetch(`${API}/projects/${projectId}/dataset/export`);
      if (!response.ok) throw new Error("Failed to download dataset");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dataset_${projectId}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);

      toast({
        title: "Download Complete",
        description: "Dataset CSV downloaded successfully"
      });

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  // ============= Test Run Handlers =============

  const handleStartTestRun = async () => {
    if (!projectId || !currentVersion) {
      toast({
        title: "Error",
        description: "Please select a prompt version first",
        variant: "destructive"
      });
      return;
    }

    if (!evalPrompt) {
      toast({
        title: "Error",
        description: "Please generate an eval prompt first",
        variant: "destructive"
      });
      return;
    }

    if (!dataset) {
      toast({
        title: "Error",
        description: "Please generate a dataset first",
        variant: "destructive"
      });
      return;
    }

    if (!isCurrentSessionDataset) {
      toast({
        title: "Error",
        description: "Please generate a new dataset for this session. The current dataset is from a previous session.",
        variant: "destructive"
      });
      return;
    }

    setIsStartingTestRun(true);
    setTestRunResults(null);
    setTestRunStatus(null);

    try {
      const versionToTest = selectedTestRunVersion || currentVersion.version;
      const response = await fetch(`${API}/projects/${projectId}/test-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_version: versionToTest,
          llm_provider: llmProvider,
          model_name: llmModel || null,
          pass_threshold: testRunPassThreshold,
          batch_size: 5,
          max_concurrent: 3
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to start test run");
      }

      const data = await response.json();
      setTestRunId(data.run_id);
      setTestRunStatus({
        status: "running",
        progress: 0,
        completed_items: 0,
        total_items: data.total_items
      });

      toast({
        title: "Test Run Started",
        description: `Running ${data.total_items} test cases...`
      });

      // Start polling for status
      pollTestRunStatus(data.run_id);

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsStartingTestRun(false);
    }
  };

  const pollTestRunStatus = async (runId) => {
    const poll = async () => {
      try {
        const response = await fetch(`${API}/projects/${projectId}/test-runs/${runId}/status`);
        if (!response.ok) return;

        const data = await response.json();
        setTestRunStatus(data);

        if (data.status === "running") {
          // Continue polling
          setTimeout(poll, 2000);
        } else if (data.status === "completed") {
          // Fetch full results
          fetchTestRunResults(runId);
          loadTestRunHistory();
          toast({
            title: "Test Run Complete",
            description: `Pass rate: ${data.partial_summary?.pass_rate || 0}%`
          });
        } else if (data.status === "failed") {
          toast({
            title: "Test Run Failed",
            description: "Check the results for error details",
            variant: "destructive"
          });
        }
      } catch (error) {
        console.error("Failed to poll status:", error);
      }
    };

    poll();
  };

  const fetchTestRunResults = async (runId) => {
    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/${runId}/results`);
      if (!response.ok) return;

      const data = await response.json();
      setTestRunResults(data);
    } catch (error) {
      console.error("Failed to fetch results:", error);
    }
  };

  const loadTestRunHistory = async () => {
    if (!projectId) return;

    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs?limit=10`);
      if (!response.ok) return;

      const data = await response.json();
      setTestRunHistory(data);
    } catch (error) {
      console.error("Failed to load test run history:", error);
    }
  };

  const handleCancelTestRun = async () => {
    if (!testRunId) return;

    try {
      await fetch(`${API}/projects/${projectId}/test-runs/${testRunId}`, {
        method: "DELETE"
      });

      setTestRunStatus(prev => ({ ...prev, status: "cancelled" }));
      toast({
        title: "Test Run Cancelled",
        description: "The test run has been stopped"
      });
    } catch (error) {
      console.error("Failed to cancel:", error);
    }
  };

  const handleViewPastRun = async (runId) => {
    setTestRunId(runId);
    await fetchTestRunResults(runId);

    // Also get status for summary
    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/${runId}/status`);
      if (response.ok) {
        const data = await response.json();
        setTestRunStatus(data);
      }
    } catch (error) {
      console.error("Failed to fetch status:", error);
    }
  };

  // Delete a test run
  const handleDeleteTestRun = async (runId, e) => {
    e.stopPropagation(); // Prevent triggering the view action

    if (!window.confirm("Are you sure you want to delete this test run? This action cannot be undone.")) {
      return;
    }

    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/${runId}/delete`, {
        method: "DELETE"
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete test run");
      }

      // Remove from history
      setTestRunHistory(prev => prev.filter(run => run.run_id !== runId));

      // Clear current view if deleted run was being viewed
      if (testRunId === runId) {
        setTestRunId(null);
        setTestRunResults(null);
        setTestRunStatus(null);
      }

      toast({
        title: "Test Run Deleted",
        description: "The test run has been permanently deleted"
      });

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  // Load test run history when project loads
  useEffect(() => {
    if (projectId && dataset && evalPrompt) {
      loadTestRunHistory();
    }
  }, [projectId, dataset, evalPrompt]);

  // Compare two test runs
  const handleCompareRuns = async () => {
    if (!compareRunA || !compareRunB) {
      toast({
        title: "Select Two Runs",
        description: "Please select both runs to compare",
        variant: "destructive"
      });
      return;
    }

    setIsComparing(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id_a: compareRunA,
          run_id_b: compareRunB
        })
      });

      if (!response.ok) throw new Error("Failed to compare runs");

      const data = await response.json();
      setComparisonResult(data);
    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsComparing(false);
    }
  };

  // Export test run results
  const handleExportResults = async (format = "csv") => {
    if (!testRunId) return;

    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/${testRunId}/export?format=${format}`);
      if (!response.ok) throw new Error("Failed to export");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `test_run_${testRunId}.${format}`;
      a.click();
      window.URL.revokeObjectURL(url);

      toast({
        title: "Export Complete",
        description: `Results exported as ${format.toUpperCase()}`
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  // Run single test
  const handleRunSingleTest = async () => {
    if (!singleTestInput.trim()) {
      toast({
        title: "Input Required",
        description: "Please enter test input",
        variant: "destructive"
      });
      return;
    }

    setIsRunningSingleTest(true);
    setSingleTestResult(null);

    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: currentVersion?.prompt_text || initialPrompt,
          test_input: { input: singleTestInput },
          llm_provider: llmProvider,
          model_name: llmModel || null,
          eval_prompt_text: evalPrompt || null
        })
      });

      if (!response.ok) throw new Error("Failed to run test");

      const data = await response.json();
      setSingleTestResult(data);
    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRunningSingleTest(false);
    }
  };

  // Re-run failed items
  const handleRerunFailed = async () => {
    if (!testRunId) return;

    setIsRerunningFailed(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/rerun-failed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_run_id: testRunId,
          llm_provider: llmProvider,
          model_name: llmModel || null
        })
      });

      if (!response.ok) throw new Error("Failed to start re-run");

      const data = await response.json();

      if (data.failed_count === 0) {
        toast({
          title: "No Failed Items",
          description: "All items passed - nothing to re-run"
        });
      } else {
        setTestRunId(data.run_id);
        setTestRunStatus({
          status: "running",
          progress: 0,
          completed_items: 0,
          total_items: data.failed_count
        });
        pollTestRunStatus(data.run_id);

        toast({
          title: "Re-run Started",
          description: `Re-running ${data.failed_count} failed items`
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRerunningFailed(false);
    }
  };

  // View result detail
  const handleViewDetail = (item) => {
    setDetailViewItem(item);
    setDetailViewOpen(true);
  };

  // Regenerate dataset with custom sample count
  const handleRegenerateDataset = async () => {
    setRegenerateDialogOpen(false);
    setDataset(null);
    setSampleCount(regenerateSampleCount);
    // Pass the count directly to avoid React state timing issues
    handleGenerateDataset(regenerateSampleCount);
  };

  const SectionHeader = ({ section, title, description }) => (
    <div
      className="flex items-center justify-between cursor-pointer p-4 bg-slate-100 dark:bg-slate-800/50 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-800/70 transition-colors"
      onClick={() => toggleSection(section)}
    >
      <div className="flex items-center gap-3">
        {expandedSections[section] ? (
          <ChevronDown className="w-5 h-5 text-blue-600 dark:text-blue-600 dark:text-blue-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-slate-600 dark:text-slate-600 dark:text-slate-400" />
        )}
        <div>
          <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
          <p className="text-sm text-slate-600 dark:text-slate-600 dark:text-slate-400">{description}</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100 to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 transition-colors">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header with Action Buttons */}
        <div className="mb-8 relative">
          {/* Top Right Buttons */}
          <div className="absolute top-0 right-0 flex gap-2">
            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Open Project Button */}
            <Dialog open={projectSelectorOpen} onOpenChange={(open) => {
              setProjectSelectorOpen(open);
              if (open) loadSavedProjects();
            }}>
              <DialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-700"
                  title="Open Project"
                >
                  <FolderOpen className="h-5 w-5" />
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-900 dark:text-white max-w-lg">
                <DialogHeader>
                  <DialogTitle className="text-xl text-slate-900 dark:text-white">Open Project</DialogTitle>
                  <DialogDescription className="text-slate-600 dark:text-slate-600 dark:text-slate-400">
                    Load a previous project or create a new one
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  {/* New Project Button */}
                  <Button
                    onClick={resetToNewProject}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Create New Project
                  </Button>

                  {/* Divider */}
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-px bg-slate-700"></div>
                    <span className="text-xs text-slate-600 dark:text-slate-500">or load existing</span>
                    <div className="flex-1 h-px bg-slate-700"></div>
                  </div>

                  {/* Projects List */}
                  <div className="max-h-[300px] overflow-y-auto space-y-2">
                    {isLoadingProjects ? (
                      <div className="text-center py-8 text-slate-600 dark:text-slate-400">
                        <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                        Loading projects...
                      </div>
                    ) : savedProjects.length === 0 ? (
                      <div className="text-center py-8 text-slate-600 dark:text-slate-400">
                        No saved projects yet
                      </div>
                    ) : (
                      savedProjects.map((project) => (
                        <div
                          key={project.id}
                          onClick={() => !isLoadingProject && loadProject(project.id)}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            projectId === project.id
                              ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-600'
                              : 'bg-slate-50 dark:bg-slate-800 border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500'
                          }`}
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <h4 className="font-medium text-slate-900 dark:text-slate-100">{project.name}</h4>
                              <p className="text-xs text-slate-600 dark:text-slate-600 dark:text-slate-400 mt-1 line-clamp-1">
                                {project.requirements?.use_case || 'No description'}
                              </p>
                            </div>
                            <div className="flex items-center gap-2 ml-2">
                              <div className="text-xs text-slate-600 dark:text-slate-500">
                                {project.system_prompt_versions?.length || 0} versions
                              </div>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={(e) => handleEditProject(project, e)}
                                className="h-6 w-6 p-0 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/20"
                                title="Edit project details"
                              >
                                <Pencil className="h-4 w-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={(e) => handleDeleteProject(project.id, project.name, e)}
                                className="h-6 w-6 p-0 text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/20"
                                title="Delete project"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                          {project.created_at && (
                            <p className="text-xs text-slate-600 dark:text-slate-500 mt-2">
                              Created: {new Date(project.created_at).toLocaleDateString()}
                            </p>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => setProjectSelectorOpen(false)}
                    className="border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    Cancel
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            {/* Settings Button */}
            <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
              <DialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-700"
                  title="Settings"
                >
                  <Settings className="h-5 w-5" />
                </Button>
              </DialogTrigger>
            <DialogContent className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-900 dark:text-white">
              <DialogHeader>
                <DialogTitle className="text-xl text-slate-900 dark:text-white">LLM Settings</DialogTitle>
                <DialogDescription className="text-slate-600 dark:text-slate-600 dark:text-slate-400">
                  Configure your LLM provider and API key for AI-powered features
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                {/* Provider Selection */}
                <div className="space-y-2">
                  <Label className="text-slate-700 dark:text-slate-300">LLM Provider</Label>
                  <Select value={llmProvider} onValueChange={(value) => {
                    setLlmProvider(value);
                    setLlmModel(""); // Reset model when provider changes
                  }}>
                    <SelectTrigger className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-white">
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                      <SelectItem value="openai" className="text-slate-900 dark:text-white hover:bg-slate-100 dark:hover:bg-slate-700">OpenAI</SelectItem>
                      <SelectItem value="claude" className="text-slate-900 dark:text-white hover:bg-slate-100 dark:hover:bg-slate-700">Claude (Anthropic)</SelectItem>
                      <SelectItem value="gemini" className="text-slate-900 dark:text-white hover:bg-slate-100 dark:hover:bg-slate-700">Gemini (Google)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Model Selection */}
                <div className="space-y-2">
                  <Label className="text-slate-700 dark:text-slate-300">Model</Label>
                  <Select value={llmModel} onValueChange={setLlmModel}>
                    <SelectTrigger className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-white">
                      <SelectValue placeholder={`Select ${llmProvider} model`} />
                    </SelectTrigger>
                    <SelectContent className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                      {modelOptions[llmProvider]?.map((model, index) => {
                        // Handle separator/disabled option objects
                        if (typeof model === 'object' && model.label) {
                          return (
                            <SelectItem key={`separator-${index}`} value={`separator-${index}`} disabled className="text-slate-500 dark:text-slate-400 font-semibold">
                              {model.label}
                            </SelectItem>
                          );
                        }
                        // Handle regular string model names
                        return (
                          <SelectItem key={model} value={model} className="text-slate-900 dark:text-white hover:bg-slate-100 dark:hover:bg-slate-700">
                            {model}
                          </SelectItem>
                        );
                      })}
                    </SelectContent>
                  </Select>
                </div>

                {/* API Key Input */}
                <div className="space-y-2">
                  <Label className="text-slate-700 dark:text-slate-300">API Key</Label>
                  <div className="relative">
                    <Input
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={`Enter your ${llmProvider === "claude" ? "Anthropic" : llmProvider === "gemini" ? "Google AI" : "OpenAI"} API key`}
                      className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-white pr-10 placeholder:text-slate-400 dark:placeholder:text-slate-500"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="absolute right-0 top-0 h-full px-3 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-white"
                      onClick={() => setShowApiKey(!showApiKey)}
                    >
                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-slate-600 dark:text-slate-500">
                    Your API key is stored locally and used for AI rewrite and evaluation features
                  </p>
                </div>

                {/* Status Indicator */}
                {settingsLoaded && apiKey && (
                  <div className="flex items-center gap-2 p-2 bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-sm text-green-600 dark:text-green-400">Settings configured</span>
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setSettingsOpen(false)}
                  className="border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSaveSettings}
                  disabled={isSavingSettings}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {isSavingSettings ? "Saving..." : "Save Settings"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          </div>

          {/* Title - Centered */}
          <div className="text-center">
            <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-600 dark:from-blue-400 dark:to-cyan-400 mb-3">
              Athena
            </h1>
            {projectId && projectName && (
              <p className="text-sm text-blue-600 dark:text-blue-600 dark:text-blue-400 mb-1">
                Current Project: <span className="font-medium">{projectName}</span>
              </p>
            )}
            <p className="text-xl text-slate-700 dark:text-slate-300 mb-2">
              Your Strategic Prompt Architect
            </p>
            <p className="text-slate-600 dark:text-slate-600 dark:text-slate-400">
              Transform your system prompts through requirements-driven analysis, AI-powered improvements, and comprehensive testing
            </p>
          </div>
        </div>

        {/* Server Status Banner */}
        {serverStatus === "offline" && (
          <div className="bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg p-4 mb-6">
            <div className="flex items-start gap-3">
              <div className="text-red-600 dark:text-red-400 text-xl">⚠️</div>
              <div className="flex-1">
                <h3 className="font-semibold text-red-600 dark:text-red-400 mb-1">Backend Server Not Connected</h3>
                <p className="text-sm text-slate-300 mb-2">
                  Cannot connect to the backend server at <code className="bg-slate-800 px-1 py-0.5 rounded">localhost:8000</code>
                </p>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                  The server may need to be restarted to load the new project API endpoints.
                </p>
                <div className="bg-slate-800 rounded p-3 text-sm font-mono">
                  <div className="text-slate-600 dark:text-slate-400 mb-1"># Stop the current server</div>
                  <div className="text-green-600 dark:text-green-400">pkill -f uvicorn</div>
                  <div className="text-slate-600 dark:text-slate-400 mt-2 mb-1"># Restart with the start script</div>
                  <div className="text-green-600 dark:text-green-400">./start.sh</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {serverStatus === "online" && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg p-3 mb-6">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-600 dark:text-green-400">Backend server connected</span>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {/* Section 1: Requirements & Initial Prompt */}
          <Card className="bg-white dark:bg-slate-900/50 border-slate-300 dark:border-slate-700">
            <CardHeader className="p-0">
              <SectionHeader
                section="requirements"
                title="1. Requirements & Initial Prompt"
                description="Define your use case and provide an initial system prompt"
              />
            </CardHeader>
            {expandedSections.requirements && (
              <CardContent className="p-6 space-y-4">
                <div>
                  <Label htmlFor="projectName">Project Name *</Label>
                  <Input
                    id="projectName"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    placeholder="e.g., Customer Support Bot"
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100"
                  />
                </div>

                <div>
                  <Label htmlFor="useCase">Use Case *</Label>
                  <Textarea
                    id="useCase"
                    value={useCase}
                    onChange={(e) => setUseCase(e.target.value)}
                    placeholder="Describe what this system prompt is for..."
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[80px]"
                  />
                </div>

                <div>
                  <Label htmlFor="requirements">Key Requirements * (one per line or comma-separated)</Label>
                  <Textarea
                    id="requirements"
                    value={keyRequirements}
                    onChange={(e) => setKeyRequirements(e.target.value)}
                    placeholder="e.g.,&#10;- Handle customer queries professionally&#10;- Provide accurate product information&#10;- Escalate complex issues to human agents"
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[120px]"
                  />
                </div>

                <div>
                  <Label htmlFor="provider">Target LLM Provider *</Label>
                  <Select value={targetProvider} onValueChange={setTargetProvider}>
                    <SelectTrigger className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai">OpenAI (GPT)</SelectItem>
                      <SelectItem value="claude">Anthropic (Claude)</SelectItem>
                      <SelectItem value="gemini">Google (Gemini)</SelectItem>
                      <SelectItem value="multi">Multi-provider</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="initialPrompt">Initial System Prompt *</Label>
                  <Textarea
                    id="initialPrompt"
                    value={initialPrompt}
                    onChange={(e) => setInitialPrompt(e.target.value)}
                    placeholder="Enter your initial system prompt here..."
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[200px] font-mono text-sm"
                  />
                </div>

                {(
                  <>
                    {isCreatingProject && (
                      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg p-4 mb-4">
                        <div className="flex items-center gap-3">
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400"></div>
                          <div>
                            <h3 className="font-semibold text-blue-600 dark:text-blue-400">Creating Your Project...</h3>
                            <p className="text-sm text-slate-300 mt-1">
                              Setting up project and analyzing your prompt against requirements and best practices.
                              This may take 20-30 seconds.
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                    <Button
                      onClick={handleCreateProject}
                      className="w-full"
                      disabled={isCreatingProject}
                    >
                      {isCreatingProject ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          {projectId ? "Updating Project..." : "Creating Project..."}
                        </>
                      ) : (
                        projectId ? "Update Project & Re-Analyze" : "Create Project & Analyze"
                      )}
                    </Button>
                  </>
                )}
              </CardContent>
            )}
          </Card>

          {/* Section 2: Prompt Optimization */}
          {projectId && (
            <Card className="bg-white dark:bg-slate-900/50 border-slate-300 dark:border-slate-700">
              <CardHeader className="p-0">
                <SectionHeader
                  section="optimization"
                  title="2. Prompt Optimization"
                  description="Analyze and improve your system prompt iteratively"
                />
              </CardHeader>
              {expandedSections.optimization && (
                <CardContent className="p-6 space-y-6">
                  {/* Analysis Results */}
                  {analysisResults && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-slate-800 p-4 rounded-lg">
                          <div className="text-sm text-slate-600 dark:text-slate-400">Overall Score</div>
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                            {analysisResults.overall_score.toFixed(1)}
                          </div>
                        </div>
                        <div className="bg-slate-800 p-4 rounded-lg">
                          <div className="text-sm text-slate-600 dark:text-slate-400">Requirements</div>
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                            {analysisResults.requirements_alignment_score.toFixed(1)}
                          </div>
                        </div>
                        <div className="bg-slate-800 p-4 rounded-lg">
                          <div className="text-sm text-slate-600 dark:text-slate-400">Best Practices</div>
                          <div className="text-3xl font-bold text-purple-400">
                            {analysisResults.best_practices_score.toFixed(1)}
                          </div>
                        </div>
                      </div>

                      {analysisResults.requirements_gaps.length > 0 && (
                        <div className="bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-lg p-4">
                          <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">Missing Requirements:</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-slate-300">
                            {analysisResults.requirements_gaps.map((gap, idx) => (
                              <li key={idx}>{gap}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {analysisResults.suggestions.length > 0 && (
                        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg p-4">
                          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Top Suggestions:</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-slate-300">
                            {analysisResults.suggestions.slice(0, 5).map((sug, idx) => (
                              <li key={idx}>
                                <span className="font-medium">[{sug.priority}]</span> {sug.suggestion}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Current Prompt */}
                  {currentVersion && (
                    <div>
                      <Label>Current Prompt (Version {currentVersion.version})</Label>
                      <Textarea
                        value={currentVersion.prompt_text}
                        readOnly
                        className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[200px] font-mono text-sm"
                      />
                    </div>
                  )}

                  {/* Changes from last refinement */}
                  {promptChanges.length > 0 && (
                    <div className="bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg p-4">
                      <h3 className="font-semibold text-green-600 dark:text-green-400 mb-2">Recent Changes</h3>
                      <ul className="text-sm text-slate-300 space-y-1">
                        {promptChanges.map((change, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-green-600 dark:text-green-400">+</span>
                            <span>{change}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Feedback/Review Section for System Prompt */}
                  {showPromptFeedback ? (
                    <div className="bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-600 rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-slate-900 dark:text-slate-200 flex items-center gap-2">
                          <MessageSquare className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                          Provide Feedback for Prompt Refinement
                        </h3>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setShowPromptFeedback(false);
                            setPromptFeedback("");
                          }}
                          className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        Describe specific changes you'd like to make to the system prompt. The AI will incorporate your feedback.
                      </p>
                      <Textarea
                        value={promptFeedback}
                        onChange={(e) => setPromptFeedback(e.target.value)}
                        placeholder="e.g., Add more specific instructions for error handling, Include examples of expected output format, Make the tone more professional..."
                        className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[100px]"
                      />
                      <div className="flex gap-3">
                        <Button
                          onClick={handleRefinePromptWithFeedback}
                          disabled={isRefiningPrompt || !promptFeedback.trim()}
                          className="bg-blue-600 hover:bg-blue-700"
                        >
                          {isRefiningPrompt ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Refining...
                            </>
                          ) : (
                            <>
                              <Send className="w-4 h-4 mr-2" />
                              Submit Feedback
                            </>
                          )}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => {
                            setShowPromptFeedback(false);
                            setPromptFeedback("");
                          }}
                          className="border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300"
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : null}

                  {/* Actions */}
                  <div className="flex gap-4 flex-wrap">
                    <Button
                      onClick={() => handleAnalyze(currentVersion?.prompt_text)}
                      disabled={isAnalyzing}
                      className="bg-slate-900 dark:bg-slate-900 text-white hover:bg-slate-800 dark:hover:bg-slate-800"
                    >
                      {isAnalyzing ? "Analyzing..." : "Re-Analyze"}
                    </Button>
                    <Button onClick={handleRewrite} disabled={isRewriting}>
                      {isRewriting ? "Rewriting..." : <><RefreshCw className="w-4 h-4 mr-2" />AI Rewrite</>}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setShowPromptFeedback(true)}
                      className="border-blue-600 text-blue-600 dark:text-blue-400 hover:bg-blue-900/20"
                    >
                      <MessageSquare className="w-4 h-4 mr-2" />
                      Review & Refine
                    </Button>
                    <Button
                      onClick={() => {
                        setExpandedSections(prev => ({
                          ...prev,
                          optimization: false,
                          evalPrompt: true
                        }));
                      }}
                      variant="secondary"
                    >
                      Continue to Eval Prompt
                    </Button>
                  </div>

                  {/* Version History */}
                  {versionHistory.length > 1 && (
                    <div className="mt-6">
                      <h3 className="font-semibold mb-2 text-slate-900 dark:text-slate-100">Version History</h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">Click on a version to load it</p>
                      <div className="space-y-2">
                        {versionHistory.map((v, idx) => (
                          <div
                            key={idx}
                            onClick={() => {
                              if (v.version !== currentVersion?.version) {
                                setCurrentVersion(v);
                                // Load analysis for this version if available
                                if (v.evaluation) {
                                  setAnalysisResults({
                                    overall_score: ((v.evaluation.requirements_alignment || 0) + (v.evaluation.best_practices_score || 0)) / 2,
                                    requirements_alignment_score: v.evaluation.requirements_alignment || 0,
                                    best_practices_score: v.evaluation.best_practices_score || 0,
                                    suggestions: v.evaluation.suggestions || [],
                                    requirements_gaps: v.evaluation.requirements_gaps || []
                                  });
                                } else {
                                  // Clear analysis and prompt re-analyze
                                  setAnalysisResults(null);
                                }
                                toast({
                                  title: "Version Loaded",
                                  description: `Switched to Version ${v.version}`
                                });
                              }
                            }}
                            className={`p-3 rounded-lg border cursor-pointer transition-all ${
                              v.version === currentVersion?.version
                                ? 'bg-blue-100 dark:bg-blue-900/20 border-blue-600 dark:border-blue-700'
                                : 'bg-slate-50 dark:bg-slate-800 border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-slate-100 dark:hover:bg-slate-700'
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-slate-900 dark:text-white">Version {v.version}</span>
                                {v.version === currentVersion?.version && (
                                  <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded">Current</span>
                                )}
                                {v.is_final && <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded">Final</span>}
                              </div>
                              <div className="flex items-center gap-2">
                                {v.evaluation && (
                                  <span className={`text-sm font-semibold px-2 py-0.5 rounded ${
                                    ((v.evaluation.requirements_alignment + v.evaluation.best_practices_score) / 2) >= 80
                                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                      : ((v.evaluation.requirements_alignment + v.evaluation.best_practices_score) / 2) >= 60
                                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                  }`}>
                                    {((v.evaluation.requirements_alignment + v.evaluation.best_practices_score) / 2).toFixed(1)}
                                  </span>
                                )}
                                {v.version !== currentVersion?.version && versionHistory.length > 1 && (
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={(e) => {
                                      e.stopPropagation(); // Prevent triggering the parent onClick
                                      handleDeleteVersion(v.version);
                                    }}
                                    className="h-6 w-6 p-0 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/20"
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              )}
            </Card>
          )}

          {/* Section 3: Evaluation Prompt */}
          {projectId && (
            <Card className="bg-white dark:bg-slate-900/50 border-slate-300 dark:border-slate-700">
              <CardHeader className="p-0">
                <SectionHeader
                  section="evalPrompt"
                  title="3. Evaluation Prompt"
                  description="Generate a prompt to evaluate your system prompt"
                />
              </CardHeader>
              {expandedSections.evalPrompt && (
                <CardContent className="p-6 space-y-4">
                  {!evalPrompt ? (
                    <Button onClick={handleGenerateEvalPrompt} disabled={isGeneratingEval} className="w-full">
                      {isGeneratingEval ? "Generating..." : "Generate Evaluation Prompt"}
                    </Button>
                  ) : (
                    <>
                      {evalRationale && (
                        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg p-4">
                          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Rationale</h3>
                          <p className="text-sm text-slate-300 whitespace-pre-line">{evalRationale}</p>
                        </div>
                      )}

                      {/* Changes from last refinement */}
                      {evalChanges.length > 0 && (
                        <div className="bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg p-4">
                          <h3 className="font-semibold text-green-600 dark:text-green-400 mb-2">Recent Changes</h3>
                          <ul className="text-sm text-slate-300 space-y-1">
                            {evalChanges.map((change, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-green-600 dark:text-green-400">+</span>
                                <span>{change}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      <div>
                        <Label>Evaluation Prompt</Label>
                        <Textarea
                          value={evalPrompt}
                          onChange={(e) => setEvalPrompt(e.target.value)}
                          className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[300px] font-mono text-sm"
                        />
                      </div>

                      {/* Feedback/Review Section */}
                      {showEvalFeedback ? (
                        <div className="bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-600 rounded-lg p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <h3 className="font-semibold text-slate-900 dark:text-slate-200 flex items-center gap-2">
                              <MessageSquare className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                              Provide Feedback for Refinement
                            </h3>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setShowEvalFeedback(false);
                                setEvalFeedback("");
                              }}
                              className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                            >
                              <X className="w-4 h-4" />
                            </Button>
                          </div>
                          <p className="text-sm text-slate-600 dark:text-slate-400">
                            Describe what changes you'd like to make to the evaluation prompt. The AI will incorporate your feedback.
                          </p>
                          <Textarea
                            value={evalFeedback}
                            onChange={(e) => setEvalFeedback(e.target.value)}
                            placeholder="e.g., Add more emphasis on code quality evaluation, Include security considerations, Make the scoring more strict..."
                            className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[100px]"
                          />
                          <div className="flex gap-3">
                            <Button
                              onClick={handleRefineEvalPrompt}
                              disabled={isRefiningEval || !evalFeedback.trim()}
                              className="bg-blue-600 hover:bg-blue-700"
                            >
                              {isRefiningEval ? (
                                <>
                                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                  Refining...
                                </>
                              ) : (
                                <>
                                  <Send className="w-4 h-4 mr-2" />
                                  Submit Feedback
                                </>
                              )}
                            </Button>
                            <Button
                              variant="outline"
                              onClick={() => {
                                setShowEvalFeedback(false);
                                setEvalFeedback("");
                              }}
                              className="border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300"
                            >
                              Cancel
                            </Button>
                          </div>
                        </div>
                      ) : null}

                      <div className="flex gap-4 flex-wrap">
                        <Button
                          variant="outline"
                          onClick={() => setShowEvalFeedback(true)}
                          className="border-blue-600 text-blue-600 dark:text-blue-400 hover:bg-blue-900/20"
                        >
                          <MessageSquare className="w-4 h-4 mr-2" />
                          Review & Refine
                        </Button>
                        <Button className="bg-slate-900 dark:bg-slate-900 text-white hover:bg-slate-800 dark:hover:bg-slate-800">
                          <Save className="w-4 h-4 mr-2" />
                          Save Changes
                        </Button>
                        <Button
                          onClick={() => {
                            setExpandedSections(prev => ({
                              ...prev,
                              evalPrompt: false,
                              dataset: true
                            }));
                          }}
                          variant="secondary"
                        >
                          Continue to Dataset
                        </Button>
                      </div>
                    </>
                  )}
                </CardContent>
              )}
            </Card>
          )}

          {/* Section 4: Test Dataset */}
          {projectId && (
            <Card className="bg-white dark:bg-slate-900/50 border-slate-300 dark:border-slate-700">
              <CardHeader className="p-0">
                <SectionHeader
                  section="dataset"
                  title="4. Test Dataset"
                  description="Generate test cases to evaluate your system prompt"
                />
              </CardHeader>
              {expandedSections.dataset && (
                <CardContent className="p-6 space-y-4">
                  {/* Sample count input */}
                  {!dataset && (
                    <div>
                      <Label htmlFor="sampleCount" className="text-slate-700 dark:text-slate-300">Number of Samples</Label>
                      <Input
                        id="sampleCount"
                        type="number"
                        value={sampleCount}
                        onChange={(e) => setSampleCount(parseInt(e.target.value))}
                        min="10"
                        max="500"
                        className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100"
                        disabled={isGeneratingDataset}
                      />
                      <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                        Default distribution: 40% positive, 30% edge cases, 20% negative, 10% adversarial
                      </p>
                    </div>
                  )}

                  {/* Progress indicator during generation */}
                  {isGeneratingDataset && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg p-4 space-y-3">
                      <div className="flex items-center gap-3">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400"></div>
                        <div className="flex-1">
                          <h3 className="font-semibold text-blue-600 dark:text-blue-400">Generating Dataset...</h3>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                            {datasetProgress.total_batches > 0
                              ? `Processing batch ${datasetProgress.batch} of ${datasetProgress.total_batches}`
                              : 'Initializing generation...'}
                          </p>
                        </div>
                      </div>

                      {/* Progress bar */}
                      <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(datasetProgress.progress, 100)}%` }}
                        ></div>
                      </div>

                      <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400">
                        <span>{Math.round(datasetProgress.progress)}% complete</span>
                        <span className="text-green-500">● Connected</span>
                      </div>
                    </div>
                  )}

                  {/* Generate button (shown when no dataset and not generating) */}
                  {!dataset && !isGeneratingDataset && (
                    <Button onClick={handleGenerateDataset} className="w-full">
                      Generate Dataset
                    </Button>
                  )}

                  {/* Dataset generated success message and preview */}
                  {dataset && (
                    <>
                      <div className="bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg p-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-2">Dataset Generated</h3>
                            <p className="text-sm text-slate-700 dark:text-slate-300">
                              {dataset.sample_count} test cases created
                            </p>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setRegenerateSampleCount(dataset?.sample_count || 100);
                              setRegenerateDialogOpen(true);
                            }}
                            disabled={isGeneratingDataset}
                            className="border-green-500 text-green-600 hover:bg-green-50 dark:border-green-600 dark:text-green-400 dark:hover:bg-green-900/30"
                          >
                            {isGeneratingDataset ? (
                              <RefreshCw className="w-4 h-4 animate-spin" />
                            ) : (
                              <>
                                <RefreshCw className="w-4 h-4 mr-1" />
                                Regenerate
                              </>
                            )}
                          </Button>
                        </div>
                      </div>

                      {/* Preview */}
                      <div>
                        <h3 className="font-semibold mb-2 text-slate-900 dark:text-slate-100">
                          Preview ({dataset.preview?.length === dataset.sample_count
                            ? `All ${dataset.sample_count} cases`
                            : `First ${dataset.preview?.length || 10} of ${dataset.sample_count} cases`})
                        </h3>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead className="bg-slate-800 dark:bg-slate-800">
                              <tr>
                                <th className="p-2 text-left text-white">Input</th>
                                <th className="p-2 text-left text-white">Category</th>
                                <th className="p-2 text-left text-white">Test Focus</th>
                                <th className="p-2 text-left text-white">Difficulty</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dataset.preview?.map((test, idx) => (
                                <tr key={idx} className="border-t border-slate-300 dark:border-slate-700">
                                  <td className="p-2 text-slate-900 dark:text-slate-100">{test.input.substring(0, 100)}...</td>
                                  <td className="p-2">
                                    <span className={`px-2 py-1 rounded text-xs text-white ${
                                      test.category === 'positive' ? 'bg-green-600' :
                                      test.category === 'edge_case' ? 'bg-yellow-600' :
                                      test.category === 'negative' ? 'bg-red-600' :
                                      'bg-purple-600'
                                    }`}>
                                      {test.category}
                                    </span>
                                  </td>
                                  <td className="p-2 text-slate-600 dark:text-slate-400">{test.test_focus}</td>
                                  <td className="p-2 text-slate-600 dark:text-slate-400">{test.difficulty}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <Button onClick={handleDownloadDataset} variant="outline" className="flex-1">
                          <Download className="w-4 h-4 mr-2" />
                          Download CSV
                        </Button>
                        {evalPrompt && (
                          <Button
                            onClick={() => {
                              // Expand test run section
                              setExpandedSections(prev => ({
                                ...prev,
                                testRun: true
                              }));
                              // Scroll to test run section after a short delay
                              setTimeout(() => {
                                document.getElementById('test-run-section')?.scrollIntoView({
                                  behavior: 'smooth',
                                  block: 'start'
                                });
                              }, 100);
                            }}
                            className="flex-1 bg-green-600 hover:bg-green-700"
                          >
                            <Play className="w-4 h-4 mr-2" />
                            Continue to Test Run
                          </Button>
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              )}
            </Card>
          )}

          {/* Section 5: Test Run */}
          {dataset && evalPrompt && (
            <Card id="test-run-section" className="border-slate-300 dark:border-slate-700 bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
              <SectionHeader
                section="testRun"
                title="5. Test Run"
                description="Execute prompt against dataset and evaluate results"
              />
              {expandedSections.testRun && (
                <CardContent className="space-y-6 pt-4">
                  {/* Warning for non-current session dataset */}
                  {!isCurrentSessionDataset && dataset && (
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700 rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                        <div>
                          <h4 className="font-semibold text-yellow-700 dark:text-yellow-300">Dataset from Previous Session</h4>
                          <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-1">
                            The current dataset was loaded from a saved project. Please generate a new dataset to run tests.
                          </p>
                          <Button
                            size="sm"
                            className="mt-2 bg-yellow-600 hover:bg-yellow-700"
                            onClick={() => {
                              setRegenerateSampleCount(dataset?.sample_count || 100);
                              setRegenerateDialogOpen(true);
                            }}
                          >
                            <RefreshCw className="w-4 h-4 mr-1" />
                            Generate New Dataset
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Configuration */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <Label className="text-slate-700 dark:text-slate-300">Prompt Version</Label>
                      <Select
                        value={String(selectedTestRunVersion || currentVersion?.version || 1)}
                        onValueChange={(v) => setSelectedTestRunVersion(parseInt(v))}
                      >
                        <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600">
                          <SelectValue placeholder="Select version" />
                        </SelectTrigger>
                        <SelectContent>
                          {versionHistory.map((v) => (
                            <SelectItem key={v.version} value={String(v.version)}>
                              Version {v.version} {v.is_final ? "(Final)" : ""}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label className="text-slate-700 dark:text-slate-300">Pass Threshold</Label>
                      <Select
                        value={String(testRunPassThreshold)}
                        onValueChange={(v) => setTestRunPassThreshold(parseFloat(v))}
                      >
                        <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="3.0">3.0 (Lenient)</SelectItem>
                          <SelectItem value="3.5">3.5 (Standard)</SelectItem>
                          <SelectItem value="4.0">4.0 (Strict)</SelectItem>
                          <SelectItem value="4.5">4.5 (Very Strict)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-end">
                      {testRunStatus?.status === "running" ? (
                        <Button
                          onClick={handleCancelTestRun}
                          variant="destructive"
                          className="w-full"
                        >
                          <Square className="w-4 h-4 mr-2" />
                          Cancel Run
                        </Button>
                      ) : (
                        <Button
                          onClick={handleStartTestRun}
                          disabled={isStartingTestRun}
                          className="w-full bg-green-600 hover:bg-green-700"
                        >
                          {isStartingTestRun ? (
                            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          ) : (
                            <Play className="w-4 h-4 mr-2" />
                          )}
                          Start Test Run
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {testRunStatus?.status === "running" && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm text-slate-600 dark:text-slate-400">
                        <span>Progress: {testRunStatus.completed_items} / {testRunStatus.total_items}</span>
                        <span>{Math.round(testRunStatus.progress)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
                        <div
                          className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                          style={{ width: `${testRunStatus.progress}%` }}
                        />
                      </div>
                      {testRunStatus.partial_summary && (
                        <div className="flex gap-4 text-sm">
                          <span className="text-green-600">
                            <CheckCircle className="w-4 h-4 inline mr-1" />
                            {testRunStatus.partial_summary.passed_items} passed
                          </span>
                          <span className="text-red-600">
                            <XCircle className="w-4 h-4 inline mr-1" />
                            {testRunStatus.partial_summary.failed_items} failed
                          </span>
                          <span className="text-slate-600 dark:text-slate-400">
                            Avg Score: {testRunStatus.partial_summary.avg_score?.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Summary Stats */}
                  {testRunResults?.summary && testRunStatus?.status !== "running" && (
                    <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4">
                      <h3 className="font-semibold mb-3 flex items-center text-slate-900 dark:text-slate-100">
                        <BarChart3 className="w-5 h-5 mr-2" />
                        Test Run Summary
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-2xl font-bold text-blue-600">
                            {testRunResults.summary.pass_rate}%
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">Pass Rate</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-2xl font-bold text-green-600">
                            {testRunResults.summary.avg_score?.toFixed(2)}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">Avg Score</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">
                            {testRunResults.summary.completed_items}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">Total Tests</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-2xl font-bold text-purple-600">
                            ${testRunResults.summary.estimated_cost?.toFixed(4)}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">Est. Cost</div>
                        </div>
                      </div>

                      {/* Score Distribution */}
                      <div className="mt-4">
                        <h4 className="text-sm font-medium mb-2 text-slate-700 dark:text-slate-300">Score Distribution</h4>
                        <div className="flex gap-2">
                          {[1, 2, 3, 4, 5].map((score) => (
                            <div key={score} className="flex-1 text-center">
                              <div className="h-16 bg-slate-200 dark:bg-slate-700 rounded relative">
                                <div
                                  className={`absolute bottom-0 w-full rounded transition-all ${
                                    score >= 4 ? 'bg-green-500' :
                                    score === 3 ? 'bg-yellow-500' :
                                    'bg-red-500'
                                  }`}
                                  style={{
                                    height: `${Math.min(100, (testRunResults.summary.score_distribution?.[String(score)] || 0) / testRunResults.summary.completed_items * 100)}%`
                                  }}
                                />
                              </div>
                              <div className="text-xs mt-1 text-slate-600 dark:text-slate-400">
                                {score} ({testRunResults.summary.score_distribution?.[String(score)] || 0})
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  {testRunResults && testRunStatus?.status !== "running" && (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleExportResults("csv")}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        Export CSV
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleExportResults("json")}
                      >
                        <FileText className="w-4 h-4 mr-1" />
                        Export JSON
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSingleTestOpen(true)}
                      >
                        <Play className="w-4 h-4 mr-1" />
                        Single Test
                      </Button>
                      {testRunResults.summary?.failed_items > 0 && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleRerunFailed}
                          disabled={isRerunningFailed}
                        >
                          {isRerunningFailed ? (
                            <RefreshCw className="w-4 h-4 mr-1 animate-spin" />
                          ) : (
                            <RotateCcw className="w-4 h-4 mr-1" />
                          )}
                          Re-run Failed ({testRunResults.summary.failed_items})
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCompareMode(!compareMode)}
                        className={compareMode ? 'bg-blue-100 dark:bg-blue-900/30' : ''}
                      >
                        <ArrowUpDown className="w-4 h-4 mr-1" />
                        Compare Runs
                      </Button>
                    </div>
                  )}

                  {/* Comparison Mode */}
                  {compareMode && testRunHistory.length >= 2 && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 space-y-4">
                      <h3 className="font-semibold text-blue-700 dark:text-blue-300">Compare Test Runs</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <Label className="text-slate-700 dark:text-slate-300">Run A (Baseline)</Label>
                          <Select value={compareRunA || ""} onValueChange={setCompareRunA}>
                            <SelectTrigger className="mt-1">
                              <SelectValue placeholder="Select run..." />
                            </SelectTrigger>
                            <SelectContent>
                              {testRunHistory.map((run) => (
                                <SelectItem key={run.run_id} value={run.run_id}>
                                  V{run.prompt_version} - {run.summary?.pass_rate || 0}% pass
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label className="text-slate-700 dark:text-slate-300">Run B (Compare)</Label>
                          <Select value={compareRunB || ""} onValueChange={setCompareRunB}>
                            <SelectTrigger className="mt-1">
                              <SelectValue placeholder="Select run..." />
                            </SelectTrigger>
                            <SelectContent>
                              {testRunHistory.map((run) => (
                                <SelectItem key={run.run_id} value={run.run_id}>
                                  V{run.prompt_version} - {run.summary?.pass_rate || 0}% pass
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="flex items-end">
                          <Button onClick={handleCompareRuns} disabled={isComparing || !compareRunA || !compareRunB}>
                            {isComparing ? <RefreshCw className="w-4 h-4 mr-1 animate-spin" /> : <ArrowUpDown className="w-4 h-4 mr-1" />}
                            Compare
                          </Button>
                        </div>
                      </div>

                      {/* Comparison Results */}
                      {comparisonResult && (
                        <div className="mt-4 space-y-4">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                              <div className="text-lg font-bold">
                                {comparisonResult.run_a.pass_rate}% → {comparisonResult.run_b.pass_rate}%
                              </div>
                              <div className={`text-sm ${comparisonResult.pass_rate_delta > 0 ? 'text-green-600' : comparisonResult.pass_rate_delta < 0 ? 'text-red-600' : 'text-slate-600'}`}>
                                {comparisonResult.pass_rate_delta > 0 ? '+' : ''}{comparisonResult.pass_rate_delta}% Pass Rate
                              </div>
                            </div>
                            <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                              <div className="text-lg font-bold">
                                {comparisonResult.run_a.avg_score?.toFixed(2)} → {comparisonResult.run_b.avg_score?.toFixed(2)}
                              </div>
                              <div className={`text-sm ${comparisonResult.avg_score_delta > 0 ? 'text-green-600' : comparisonResult.avg_score_delta < 0 ? 'text-red-600' : 'text-slate-600'}`}>
                                {comparisonResult.avg_score_delta > 0 ? '+' : ''}{comparisonResult.avg_score_delta} Avg Score
                              </div>
                            </div>
                            <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                              <div className="flex justify-center gap-2">
                                <span className="text-green-600 flex items-center"><TrendingUp className="w-4 h-4 mr-1" />{comparisonResult.improved_items}</span>
                                <span className="text-red-600 flex items-center"><TrendingDown className="w-4 h-4 mr-1" />{comparisonResult.regressed_items}</span>
                              </div>
                              <div className="text-sm text-slate-600 dark:text-slate-400">Improved / Regressed</div>
                            </div>
                            <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg">
                              <div className="text-lg font-bold text-slate-600">{comparisonResult.unchanged_items}</div>
                              <div className="text-sm text-slate-600 dark:text-slate-400">Unchanged</div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Results Table */}
                  {testRunResults?.results && testRunResults.results.length > 0 && (
                    <div>
                      <h3 className="font-semibold mb-2 text-slate-900 dark:text-slate-100">
                        Detailed Results ({testRunResults.results.length} items)
                        <span className="text-sm font-normal text-slate-500 ml-2">Click row for details</span>
                      </h3>
                      <div className="overflow-x-auto max-h-96 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-slate-800 dark:bg-slate-800 sticky top-0">
                            <tr>
                              <th className="p-2 text-left text-white w-8">#</th>
                              <th className="p-2 text-left text-white">Input</th>
                              <th className="p-2 text-left text-white">Output</th>
                              <th className="p-2 text-left text-white w-20">Score</th>
                              <th className="p-2 text-left text-white w-20">Status</th>
                              <th className="p-2 text-left text-white">Feedback</th>
                            </tr>
                          </thead>
                          <tbody>
                            {testRunResults.results.map((result, idx) => (
                              <tr
                                key={result.id || idx}
                                onClick={() => handleViewDetail(result)}
                                className={`border-t border-slate-300 dark:border-slate-700 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 ${
                                  result.passed
                                    ? 'bg-green-50 dark:bg-green-900/10'
                                    : 'bg-red-50 dark:bg-red-900/10'
                                }`}
                              >
                                <td className="p-2 text-slate-600 dark:text-slate-400">
                                  {result.dataset_item_index + 1}
                                </td>
                                <td className="p-2 text-slate-900 dark:text-slate-100 max-w-xs truncate">
                                  {result.input_data?.input?.substring(0, 80) || JSON.stringify(result.input_data).substring(0, 80)}...
                                </td>
                                <td className="p-2 text-slate-900 dark:text-slate-100 max-w-xs truncate">
                                  {result.prompt_output?.substring(0, 100)}...
                                </td>
                                <td className="p-2">
                                  <span className={`px-2 py-1 rounded text-xs font-bold ${
                                    result.eval_score >= 4 ? 'bg-green-600 text-white' :
                                    result.eval_score >= 3 ? 'bg-yellow-600 text-white' :
                                    'bg-red-600 text-white'
                                  }`}>
                                    {result.eval_score?.toFixed(1)}
                                  </span>
                                </td>
                                <td className="p-2">
                                  {result.error ? (
                                    <span className="text-red-600 flex items-center">
                                      <AlertCircle className="w-4 h-4 mr-1" />
                                      Error
                                    </span>
                                  ) : result.passed ? (
                                    <span className="text-green-600 flex items-center">
                                      <CheckCircle className="w-4 h-4 mr-1" />
                                      Pass
                                    </span>
                                  ) : (
                                    <span className="text-red-600 flex items-center">
                                      <XCircle className="w-4 h-4 mr-1" />
                                      Fail
                                    </span>
                                  )}
                                </td>
                                <td className="p-2 text-slate-600 dark:text-slate-400 max-w-md truncate">
                                  {result.eval_feedback?.substring(0, 150)}...
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Test Run History */}
                  {testRunHistory.length > 0 && (
                    <div>
                      <h3 className="font-semibold mb-2 text-slate-900 dark:text-slate-100">
                        Previous Test Runs
                      </h3>
                      <div className="space-y-2">
                        {testRunHistory.map((run) => (
                          <div
                            key={run.run_id}
                            className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                              testRunId === run.run_id
                                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                                : 'border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-800'
                            }`}
                            onClick={() => handleViewPastRun(run.run_id)}
                          >
                            <div className="flex justify-between items-center">
                              <div>
                                <span className="font-medium text-slate-900 dark:text-slate-100">
                                  Version {run.prompt_version}
                                </span>
                                <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                                  run.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                                  run.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                                  'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300'
                                }`}>
                                  {run.status}
                                </span>
                              </div>
                              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                                {run.summary && (
                                  <span>
                                    Pass: {run.summary.pass_rate}%
                                  </span>
                                )}
                                <span>{new Date(run.created_at).toLocaleDateString()}</span>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-7 w-7 p-0 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20"
                                  onClick={(e) => handleDeleteTestRun(run.run_id, e)}
                                  title="Delete test run"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              )}
            </Card>
          )}
        </div>
      </div>

      {/* Single Test Modal */}
      <Dialog open={singleTestOpen} onOpenChange={setSingleTestOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Single Test Mode</DialogTitle>
            <DialogDescription>
              Test your prompt with a single input to debug and verify behavior
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label>Test Input</Label>
              <Textarea
                value={singleTestInput}
                onChange={(e) => setSingleTestInput(e.target.value)}
                placeholder="Enter a test input message..."
                className="mt-1 min-h-24"
              />
            </div>

            <Button
              onClick={handleRunSingleTest}
              disabled={isRunningSingleTest || !singleTestInput.trim()}
              className="w-full"
            >
              {isRunningSingleTest ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run Test
                </>
              )}
            </Button>

            {singleTestResult && (
              <div className="space-y-4 border-t pt-4">
                <div>
                  <Label className="text-slate-700 dark:text-slate-300">Output</Label>
                  <div className="mt-1 p-3 bg-slate-100 dark:bg-slate-800 rounded-lg whitespace-pre-wrap text-sm max-h-48 overflow-y-auto">
                    {singleTestResult.prompt_output}
                  </div>
                </div>

                {singleTestResult.eval_score !== null && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-slate-700 dark:text-slate-300">Score</Label>
                      <div className="mt-1">
                        <span className={`px-3 py-1 rounded text-lg font-bold ${
                          singleTestResult.eval_score >= 4 ? 'bg-green-600 text-white' :
                          singleTestResult.eval_score >= 3 ? 'bg-yellow-600 text-white' :
                          'bg-red-600 text-white'
                        }`}>
                          {singleTestResult.eval_score?.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <div>
                      <Label className="text-slate-700 dark:text-slate-300">Latency</Label>
                      <div className="mt-1 text-lg">{singleTestResult.latency_ms}ms</div>
                    </div>
                  </div>
                )}

                {singleTestResult.eval_feedback && (
                  <div>
                    <Label className="text-slate-700 dark:text-slate-300">Evaluation Feedback</Label>
                    <div className="mt-1 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm whitespace-pre-wrap max-h-48 overflow-y-auto">
                      {singleTestResult.eval_feedback}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Detail View Modal */}
      <Dialog open={detailViewOpen} onOpenChange={setDetailViewOpen}>
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              Test Result #{detailViewItem?.dataset_item_index + 1}
              {detailViewItem?.passed ? (
                <span className="px-2 py-0.5 bg-green-600 text-white text-xs rounded">PASS</span>
              ) : (
                <span className="px-2 py-0.5 bg-red-600 text-white text-xs rounded">FAIL</span>
              )}
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                detailViewItem?.eval_score >= 4 ? 'bg-green-600 text-white' :
                detailViewItem?.eval_score >= 3 ? 'bg-yellow-600 text-white' :
                'bg-red-600 text-white'
              }`}>
                Score: {detailViewItem?.eval_score?.toFixed(1)}
              </span>
            </DialogTitle>
          </DialogHeader>

          {detailViewItem && (
            <div className="space-y-4">
              <div>
                <Label className="text-slate-700 dark:text-slate-300 font-semibold">Input</Label>
                <div className="mt-1 p-3 bg-slate-100 dark:bg-slate-800 rounded-lg whitespace-pre-wrap text-sm max-h-32 overflow-y-auto">
                  {detailViewItem.input_data?.input || JSON.stringify(detailViewItem.input_data, null, 2)}
                </div>
              </div>

              <div>
                <Label className="text-slate-700 dark:text-slate-300 font-semibold">Prompt Output</Label>
                <div className="mt-1 p-3 bg-slate-100 dark:bg-slate-800 rounded-lg whitespace-pre-wrap text-sm max-h-48 overflow-y-auto">
                  {detailViewItem.prompt_output}
                </div>
              </div>

              <div>
                <Label className="text-slate-700 dark:text-slate-300 font-semibold">Evaluation Feedback</Label>
                <div className="mt-1 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg whitespace-pre-wrap text-sm max-h-48 overflow-y-auto">
                  {detailViewItem.eval_feedback}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 pt-2 border-t">
                <div>
                  <Label className="text-slate-500 text-xs">Latency</Label>
                  <div className="font-mono">{detailViewItem.latency_ms}ms</div>
                </div>
                <div>
                  <Label className="text-slate-500 text-xs">Tokens Used</Label>
                  <div className="font-mono">{detailViewItem.tokens_used || 'N/A'}</div>
                </div>
                <div>
                  <Label className="text-slate-500 text-xs">Error</Label>
                  <div className={detailViewItem.error ? 'text-red-600' : 'text-green-600'}>
                    {detailViewItem.error || 'None'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Regenerate Dataset Dialog */}
      <Dialog open={regenerateDialogOpen} onOpenChange={setRegenerateDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Regenerate Dataset</DialogTitle>
            <DialogDescription>
              Choose how many test cases to generate. More cases provide better coverage but take longer.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div>
              <Label className="text-slate-700 dark:text-slate-300">Number of Test Cases</Label>
              <Select
                value={String(regenerateSampleCount)}
                onValueChange={(v) => setRegenerateSampleCount(parseInt(v))}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10 (Quick test)</SelectItem>
                  <SelectItem value="25">25 (Small)</SelectItem>
                  <SelectItem value="50">50 (Medium)</SelectItem>
                  <SelectItem value="100">100 (Standard)</SelectItem>
                  <SelectItem value="200">200 (Large)</SelectItem>
                  <SelectItem value="500">500 (Comprehensive)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-slate-500 mt-2">
                Estimated time: ~{Math.ceil(regenerateSampleCount / 10)} minutes
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setRegenerateDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleRegenerateDataset}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Generate {regenerateSampleCount} Cases
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PromptOptimizer;
