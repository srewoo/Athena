import React, { useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Save, Download, RefreshCw, Settings, Eye, EyeOff, MessageSquare, Send, X, FolderOpen, Plus, Trash2, Pencil } from "lucide-react";
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
    dataset: false
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
  const [sampleCount, setSampleCount] = useState(100);
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false);
  const [datasetProgress, setDatasetProgress] = useState({ progress: 0, batch: 0, total_batches: 0, status: '' });
  const [serverStatus, setServerStatus] = useState("checking"); // checking, online, offline

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

      // Set dataset if exists
      if (project.dataset) {
        setDataset(project.dataset);
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

  const handleGenerateDataset = async () => {
    if (!projectId) return;

    setIsGeneratingDataset(true);
    setDatasetProgress({ progress: 0, batch: 0, total_batches: 0, status: 'starting' });

    try {
      // Use streaming endpoint for large datasets with heartbeat
      const response = await fetch(`${API}/projects/${projectId}/dataset/generate-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_count: sampleCount
        })
      });

      if (!response.ok) {
        // Fall back to regular endpoint if streaming fails
        const fallbackResponse = await fetch(`${API}/projects/${projectId}/dataset/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_count: sampleCount
          })
        });

        if (!fallbackResponse.ok) throw new Error("Failed to generate dataset");

        const result = await fallbackResponse.json();
        setDataset(result);
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
                              setDataset(null);
                              handleGenerateDataset();
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
                        <h3 className="font-semibold mb-2 text-slate-900 dark:text-slate-100">Preview (First 10 cases)</h3>
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

                      <Button onClick={handleDownloadDataset} className="w-full">
                        <Download className="w-4 h-4 mr-2" />
                        Download CSV
                      </Button>
                    </>
                  )}
                </CardContent>
              )}
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default PromptOptimizer;
