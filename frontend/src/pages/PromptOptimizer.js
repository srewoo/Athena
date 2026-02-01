import React, { useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Save, Download, RefreshCw, Settings, Eye, EyeOff, MessageSquare, Send, X, FolderOpen, Plus, Trash2, Pencil, Play, PlayCircle, Square, CheckCircle, XCircle, AlertCircle, AlertTriangle, BarChart3, ArrowUpDown, RotateCcw, FileText, Maximize2, TrendingUp, TrendingDown, Minus, Sparkles, Upload } from "lucide-react";
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { useToast } from "../hooks/use-toast";
import { API, BASE_URL } from "../App";
import { ThemeToggle } from "../components/theme-toggle";
import SettingsModal from "../components/optimizer/SettingsModal";

// Helper function to safely get display text from test case input
const getTestInputDisplay = (test, maxLength = 100) => {
  try {
    if (!test) return '';
    // Handle string input
    if (typeof test.input === 'string') return test.input.substring(0, maxLength);
    // Handle inputs object (new multi-variable format)
    if (test.inputs && typeof test.inputs === 'object') {
      return JSON.stringify(test.inputs).substring(0, maxLength);
    }
    // Handle input as object
    if (test.input && typeof test.input === 'object') {
      return JSON.stringify(test.input).substring(0, maxLength);
    }
    // Fallback: try to find any variable field
    const knownFields = ['id', 'category', 'test_focus', 'expected_behavior', 'expected_output', 'difficulty', 'created_at', 'inputs', 'quality'];
    const varFields = Object.keys(test).filter(k => !knownFields.includes(k));
    if (varFields.length > 0) {
      const varsObj = {};
      varFields.forEach(k => varsObj[k] = test[k]);
      return JSON.stringify(varsObj).substring(0, maxLength);
    }
    return '';
  } catch (e) {
    return '';
  }
};

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
  
  // PRD & Enhanced Context state
  const [showEnhancedContext, setShowEnhancedContext] = useState(false);
  const [prdDocument, setPrdDocument] = useState("");
  const [tone, setTone] = useState("");
  const [outputFormat, setOutputFormat] = useState("");
  const [successCriteria, setSuccessCriteria] = useState("");
  const [edgeCases, setEdgeCases] = useState("");
  
  const [targetProvider, setTargetProvider] = useState("openai");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [projectId, setProjectId] = useState(null);
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  // Auto-extraction from system prompt
  const [isExtractingFromPrompt, setIsExtractingFromPrompt] = useState(false);
  const extractionTimeoutRef = React.useRef(null);

  // Project Management
  const [savedProjects, setSavedProjects] = useState([]);
  const [projectSelectorOpen, setProjectSelectorOpen] = useState(false);
  const [isLoadingProjects, setIsLoadingProjects] = useState(false);
  const [isLoadingProject, setIsLoadingProject] = useState(false);

  // Import Eval Prompt
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importedEvalPrompt, setImportedEvalPrompt] = useState("");
  const [importProjectName, setImportProjectName] = useState("");
  const [importDescription, setImportDescription] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const fileInputRef = React.useRef(null);

  // Section 2: Optimization
  const [analysisResults, setAnalysisResults] = useState(null);
  const [currentVersion, setCurrentVersion] = useState(null);
  const [versionHistory, setVersionHistory] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAgenticRewriting, setIsAgenticRewriting] = useState(false);
  const [agenticDetails, setAgenticDetails] = useState(null);
  const [showPromptFeedback, setShowPromptFeedback] = useState(false);
  const [promptFeedback, setPromptFeedback] = useState("");
  const [isRefiningPrompt, setIsRefiningPrompt] = useState(false);
  const [promptChanges, setPromptChanges] = useState([]);

  // Section 3: Eval Prompt
  const [evalPrompt, setEvalPrompt] = useState("");
  const [evalRationale, setEvalRationale] = useState("");
  const [isAgenticGeneratingEval, setIsAgenticGeneratingEval] = useState(false);
  const [agenticEvalDetails, setAgenticEvalDetails] = useState(null);
  const [showEvalFeedback, setShowEvalFeedback] = useState(false);
  const [evalFeedback, setEvalFeedback] = useState("");
  const [isRefiningEval, setIsRefiningEval] = useState(false);
  const [evalChanges, setEvalChanges] = useState([]);

  // Multi-Aspect Eval Generation
  const [isEvalAspectsModalOpen, setIsEvalAspectsModalOpen] = useState(false);
  const [evaluationAspects, setEvaluationAspects] = useState("");
  const [multipleEvalPrompts, setMultipleEvalPrompts] = useState([]);
  const [expandedEvalPrompts, setExpandedEvalPrompts] = useState(new Set([0])); // First prompt expanded by default
  const [isFetchingSuggestions, setIsFetchingSuggestions] = useState(false);
  const [suggestedDomain, setSuggestedDomain] = useState("");
  const [suggestionReasoning, setSuggestionReasoning] = useState("");

  // Eval Prompt Testing
  const [showEvalTest, setShowEvalTest] = useState(false);
  const [evalTestInput, setEvalTestInput] = useState("");
  const [evalTestOutput, setEvalTestOutput] = useState("");
  const [evalTestExpectedScore, setEvalTestExpectedScore] = useState("");
  const [evalTestResult, setEvalTestResult] = useState(null);
  const [isTestingEval, setIsTestingEval] = useState(false);

  // Eval Prompt Versioning
  const [evalPromptVersions, setEvalPromptVersions] = useState([]);
  const [currentEvalVersion, setCurrentEvalVersion] = useState(null);
  const [selectedEvalVersionsForCompare, setSelectedEvalVersionsForCompare] = useState([]);
  const [evalVersionCompareModalOpen, setEvalVersionCompareModalOpen] = useState(false);
  const [evalVersionDiffData, setEvalVersionDiffData] = useState(null);
  const [isLoadingEvalDiff, setIsLoadingEvalDiff] = useState(false);

  // Section 4: Dataset
  const [dataset, setDataset] = useState(null);
  const [isCurrentSessionDataset, setIsCurrentSessionDataset] = useState(false); // Track if dataset is from current session
  const [sampleCount, setSampleCount] = useState(100);
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false);
  const [datasetProgress, setDatasetProgress] = useState({ progress: 0, batch: 0, total_batches: 0, status: '' });
  const [serverStatus, setServerStatus] = useState("checking"); // checking, online, offline
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const csvUploadRef = React.useRef(null);

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

  // Regenerate states for each step
  const [isRegeneratingAnalysis, setIsRegeneratingAnalysis] = useState(false);
  const [isRegeneratingPrompt, setIsRegeneratingPrompt] = useState(false);
  const [isRegeneratingEval, setIsRegeneratingEval] = useState(false);
  const [isRegeneratingDataset, setIsRegeneratingDataset] = useState(false);

  // Settings Modal
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [llmProvider, setLlmProvider] = useState("openai");
  const [llmModel, setLlmModel] = useState("gpt-4o-mini");
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

  // State for separate evaluation model
  const [evalProvider, setEvalProvider] = useState("openai");
  const [evalModel, setEvalModel] = useState("o3-mini");  // Default to thinking model
  const [useSeparateEvalModel, setUseSeparateEvalModel] = useState(true);  // Enable by default

  // Thinking/Reasoning models for evaluation
  const thinkingModels = {
    openai: ["o1", "o1-preview", "o1-mini", "o3", "o3-mini"],
    claude: ["claude-sonnet-4-5-20250929", "claude-3-7-sonnet-20250219"],
    gemini: ["gemini-2.5-pro", "gemini-2.5-flash"]
  };

  // Version comparison state
  const [selectedVersionsForCompare, setSelectedVersionsForCompare] = useState([]);
  const [versionCompareModalOpen, setVersionCompareModalOpen] = useState(false);
  const [diffViewMode, setDiffViewMode] = useState("side-by-side"); // "side-by-side" or "unified"
  const [versionDiffData, setVersionDiffData] = useState(null);
  const [isLoadingDiff, setIsLoadingDiff] = useState(false);

  // Regression detection state
  const [regressionAlert, setRegressionAlert] = useState(null);
  const [isCheckingRegression, setIsCheckingRegression] = useState(false);

  // Load existing settings on mount - from localStorage first, then sync to backend
  useEffect(() => {
    const loadSettings = async () => {
      try {
        // First, check localStorage for saved settings (persists across browser close/refresh)
        const savedSettings = localStorage.getItem('athena_llm_settings');
        if (savedSettings) {
          const localData = JSON.parse(savedSettings);
          setLlmProvider(localData.llm_provider || "openai");
          setLlmModel(localData.model_name || "");
          setApiKey(localData.api_key || "");
          setSettingsLoaded(true);

          // Sync localStorage settings to backend (in case server restarted)
          if (localData.api_key) {
            await fetch(`${API}/settings`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(localData)
            });
            console.log("Settings synced from localStorage to backend");
          }
          return;
        }

        // Fallback: Try to load from backend
        const response = await fetch(`${API}/settings`);
        if (response.ok) {
          const data = await response.json();
          if (data && data.api_key) {
            setLlmProvider(data.llm_provider || "openai");
            setLlmModel(data.model_name || "");
            setApiKey(data.api_key || "");
            setSettingsLoaded(true);
            // Save to localStorage for persistence
            localStorage.setItem('athena_llm_settings', JSON.stringify(data));
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
    const settingsData = {
      llm_provider: llmProvider,
      api_key: apiKey,
      model_name: llmModel || modelOptions[llmProvider][0]
    };

    try {
      const response = await fetch(`${API}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settingsData)
      });

      if (response.ok) {
        // Save to localStorage for persistence across browser close/refresh/server restart
        localStorage.setItem('athena_llm_settings', JSON.stringify(settingsData));

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

  // Extract use case and key requirements from system prompt
  const extractFromPrompt = async (promptText) => {
    if (!promptText || promptText.trim().length < 20) {
      return;
    }

    // Don't extract if user has already filled in use case or requirements
    if (useCase.trim() || keyRequirements.trim()) {
      return;
    }

    setIsExtractingFromPrompt(true);
    try {
      const response = await fetch(`${API}/projects/extract-from-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ system_prompt: promptText })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.use_case && !useCase.trim()) {
          setUseCase(data.use_case);
        }
        if (data.key_requirements && data.key_requirements.length > 0 && !keyRequirements.trim()) {
          setKeyRequirements(data.key_requirements.join("\n"));
        }
        if (data.error) {
          console.warn("Extraction warning:", data.error);
        }
      }
    } catch (error) {
      console.error("Failed to extract from prompt:", error);
    } finally {
      setIsExtractingFromPrompt(false);
    }
  };

  // Handle system prompt change with debounced extraction
  const handleInitialPromptChange = (value) => {
    setInitialPrompt(value);

    // Clear any pending extraction
    if (extractionTimeoutRef.current) {
      clearTimeout(extractionTimeoutRef.current);
    }

    // Only trigger extraction if use case and requirements are empty
    if (!useCase.trim() && !keyRequirements.trim() && value.trim().length >= 50) {
      // Debounce: wait 1.5 seconds after user stops typing
      extractionTimeoutRef.current = setTimeout(() => {
        extractFromPrompt(value);
      }, 1500);
    }
  };

  // Cleanup timeout on unmount
  React.useEffect(() => {
    return () => {
      if (extractionTimeoutRef.current) {
        clearTimeout(extractionTimeoutRef.current);
      }
    };
  }, []);

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

      // Set project data (backend uses project_name, not name)
      setProjectId(project.id);
      setProjectName(project.project_name || project.name);
      setUseCase(project.requirements?.use_case || project.use_case);
      setKeyRequirements(project.requirements.key_requirements.join("\n"));
      setTargetProvider(project.requirements.target_provider);
      
      // Load enhanced context if available
      if (project.structured_requirements) {
        const sr = project.structured_requirements;
        setPrdDocument(sr.prd_document || '');
        setTone(sr.tone || '');
        setOutputFormat(sr.output_format || '');
        setSuccessCriteria(sr.success_criteria ? sr.success_criteria.join(', ') : '');
        setEdgeCases(sr.edge_cases ? sr.edge_cases.join(', ') : '');
        // Auto-expand if there are any enhanced context fields
        if (sr.prd_document || sr.tone || sr.output_format || sr.success_criteria || sr.edge_cases) {
          setShowEnhancedContext(true);
        }
      }

      // Set versions
      if (project.system_prompt_versions && project.system_prompt_versions.length > 0) {
        setVersionHistory(project.system_prompt_versions);
        const latestVersion = project.system_prompt_versions[project.system_prompt_versions.length - 1];
        setCurrentVersion(latestVersion);
        setInitialPrompt(project.system_prompt_versions[0].prompt_text);
      }

      // Set eval prompt if exists (handle multiple formats)
      if (project.eval_prompt) {
        if (typeof project.eval_prompt === 'string') {
          // String format - could be plain text or JSON string
          setEvalPrompt(project.eval_prompt);
          setEvalRationale(project.eval_rationale || '');
        } else if (project.eval_prompt.prompt_text) {
          // Object format with prompt_text field
          setEvalPrompt(project.eval_prompt.prompt_text);
          setEvalRationale(project.eval_prompt.rationale || '');
        }
      }

      // Load eval prompt versions if exists
      if (project.eval_prompt_versions && project.eval_prompt_versions.length > 0) {
        setEvalPromptVersions(project.eval_prompt_versions);
        // Find current version
        const current = project.eval_prompt_versions.find(v =>
          v.eval_prompt_text === project.eval_prompt
        );
        if (current) {
          setCurrentEvalVersion(current);
        }
      } else if (project.eval_prompt) {
        // If there's an eval prompt but no versions, fetch them from the server
        // (versions might have been created but not included in the project response)
        setTimeout(() => fetchEvalPromptVersions(), 100);
      }

      // Set dataset if exists (mark as current session since it was persisted)
      if (project.dataset) {
        setDataset(project.dataset);
        setIsCurrentSessionDataset(true); // Persisted dataset can be used for test runs
      }

      // Load test run history if exists
      if (project.test_runs && project.test_runs.length > 0) {
        // Transform to match expected format if needed
        const formattedRuns = project.test_runs.map(run => ({
          run_id: run.id,
          created_at: run.created_at,
          version_number: run.version_number,
          status: run.status,
          summary: run.summary
        }));
        setTestRunHistory(formattedRuns);
      }

      // Expand sections based on what data exists
      setExpandedSections({
        requirements: false,
        optimization: true,
        evalPrompt: !!project.eval_prompt,
        dataset: !!project.dataset,
        testRun: !!(project.test_runs && project.test_runs.length > 0)
      });

      setProjectSelectorOpen(false);

      toast({
        title: "Project Loaded",
        description: `Loaded "${project.project_name || project.name}"`
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
    setEvalPromptVersions([]);
    setCurrentEvalVersion(null);
    setSelectedEvalVersionsForCompare([]);
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

  // Handle Import Eval Prompt button click
  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  // Handle file selection
  const handleFileSelected = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file size (max 50KB)
    if (file.size > 50 * 1024) {
      toast({
        title: "File Too Large",
        description: "Please select a file smaller than 50KB",
        variant: "destructive"
      });
      event.target.value = "";
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result;
      if (!content || content.trim().length === 0) {
        toast({
          title: "Empty File",
          description: "The selected file is empty",
          variant: "destructive"
        });
        return;
      }
      setImportedEvalPrompt(content);
      setImportProjectName("");
      setImportDescription("");
      setShowImportDialog(true);
    };
    reader.onerror = () => {
      toast({
        title: "Error Reading File",
        description: "Failed to read the selected file",
        variant: "destructive"
      });
    };
    reader.readAsText(file);
    event.target.value = ""; // Reset input
  };

  // Handle creating imported eval project
  const handleCreateImportedProject = async () => {
    if (!importProjectName.trim() || !importDescription.trim()) {
      toast({
        title: "Missing Fields",
        description: "Please fill in both project name and description",
        variant: "destructive"
      });
      return;
    }

    setIsImporting(true);
    try {
      const response = await fetch(`${API}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: importProjectName.trim(),
          use_case: importDescription.trim(),
          key_requirements: [],
          initial_prompt: "",
          eval_prompt: importedEvalPrompt,
          project_type: "eval"
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to create project");
      }

      const project = await response.json();

      // Set project state
      setProjectId(project.id);
      setProjectName(project.project_name);
      setUseCase(project.use_case);
      setKeyRequirements("");
      setInitialPrompt("");
      setVersionHistory(project.system_prompt_versions || []);
      setCurrentVersion(project.system_prompt_versions?.[0] || null);
      setEvalPrompt(importedEvalPrompt);
      setEvalRationale("");

      // Set eval prompt versions from the created project
      if (project.eval_prompt_versions && project.eval_prompt_versions.length > 0) {
        setEvalPromptVersions(project.eval_prompt_versions);
        const current = project.eval_prompt_versions.find(v => v.eval_prompt_text === importedEvalPrompt);
        if (current) {
          setCurrentEvalVersion(current);
        }
      }

      // Clear import state
      setShowImportDialog(false);
      setImportedEvalPrompt("");
      setImportProjectName("");
      setImportDescription("");
      setProjectSelectorOpen(false);

      // Expand eval prompt section (skip to section 3)
      setExpandedSections({
        requirements: false,
        optimization: false,
        evalPrompt: true,
        dataset: false,
        testRun: false
      });

      toast({
        title: "Eval Prompt Imported",
        description: `Project "${project.project_name}" created with imported eval prompt`
      });

    } catch (error) {
      console.error("Import error:", error);
      toast({
        title: "Import Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsImporting(false);
    }
  };

  const handleEditProject = async (project, event) => {
    // Stop event propagation to prevent loading the project when clicking edit
    event.stopPropagation();

    // Load the project into the form fields for editing
    setProjectId(project.id);
    setProjectName(project.project_name || project.name);
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
      description: `You can now edit "${project.project_name || project.name}" and save your changes`
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
      // Parse requirements (newline separated — commas are allowed within a requirement)
      const reqList = keyRequirements
        .split(/\n/)
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

      // Build structured requirements if any fields are filled
      let structuredRequirements = null;
      if (prdDocument || tone || outputFormat || successCriteria || edgeCases) {
        structuredRequirements = {};
        
        if (prdDocument) {
          structuredRequirements.prd_document = prdDocument.trim();
        }
        if (tone) {
          structuredRequirements.tone = tone.trim();
        }
        if (outputFormat) {
          structuredRequirements.output_format = outputFormat.trim();
        }
        if (successCriteria) {
          structuredRequirements.success_criteria = successCriteria.split(',').map(s => s.trim()).filter(s => s);
        }
        if (edgeCases) {
          structuredRequirements.edge_cases = edgeCases.split(',').map(s => s.trim()).filter(s => s);
        }
      }

      const response = await fetch(`${API}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: projectName,
          use_case: useCase,
          key_requirements: reqList,
          structured_requirements: structuredRequirements,
          target_provider: targetProvider,
          initial_prompt: initialPrompt
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        let errorMessage = "Failed to create project. Please ensure the backend server is running.";
        if (errorData.detail) {
          if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          } else if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail
              .map(e => e.msg || e.message || JSON.stringify(e))
              .join('; ');
          }
        }
        throw new Error(errorMessage);
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
          overall_score: results.overall_score || 0,
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

        // Persist evaluation to backend so it survives page reload
        try {
          await fetch(`${API}/projects/${projectId}/versions/${versionToUpdate.version}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ evaluation })
          });
        } catch (saveError) {
          console.error('Failed to persist evaluation:', saveError);
          // Don't show error to user - local state still has the data
        }
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

  const handleAgenticRewrite = async () => {
    if (!projectId || !currentVersion) return;

    setIsAgenticRewriting(true);
    setAgenticDetails(null);

    try {
      // Build analysis context from Re-Analyze results
      const analysisContext = {
        score: analysisResults?.score || 0,
        suggestions: analysisResults?.suggestions || [],
        missing_requirements: analysisResults?.requirements_gaps || [],
        best_practices_gaps: analysisResults?.best_practices_gaps || [],
        categories: analysisResults?.categories || {}
      };

      const response = await fetch(`${API}/step2/agentic-rewrite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: {
            provider: llmProvider,
            api_key: apiKey,
            model_name: llmModel,
            project_name: projectName,
            use_case: useCase,
            requirements: Array.isArray(keyRequirements) ? keyRequirements.join("\n") : keyRequirements,
            initial_prompt: currentVersion.prompt_text
          },
          current_result: {
            optimized_prompt: currentVersion.prompt_text,
            score: analysisResults?.score || 0,
            analysis_context: analysisContext
          },
          use_thinking_model: true
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        // Handle FastAPI validation errors (array) and regular errors (string)
        const errorMessage = Array.isArray(errorData.detail)
          ? errorData.detail.map(e => e.msg || e.message || JSON.stringify(e)).join(', ')
          : (errorData.detail || "Agentic rewrite failed");
        throw new Error(errorMessage);
      }

      const result = await response.json();

      // Store agentic details for display
      setAgenticDetails(result.agentic_details);

      if (result.no_change) {
        toast({
          title: "No Changes Needed",
          description: result.analysis || "Prompt is already well-optimized."
        });
      } else {
        // Add new version with the improved prompt
        const changesDescription = result.improvements?.length > 0
          ? `Agentic improvements: ${result.improvements.join("; ")}`
          : "Agentic optimization applied";

        // Add version WITHOUT precomputed analysis - let addVersion trigger real analysis
        // This ensures the version tile shows accurate score from the analyze API
        await addVersion(result.optimized_prompt, changesDescription, false, null);

        const qualityDelta = result.agentic_details?.quality_delta || 0;
        toast({
          title: "Agentic Rewrite Complete",
          description: `Prompt rewritten. Analyzing quality...`
        });
      }

    } catch (error) {
      toast({
        title: "Agentic Rewrite Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsAgenticRewriting(false);
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

  const addVersion = async (promptText, feedback = "", skipAnalysis = false, precomputedAnalysis = null) => {
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

      // If we have precomputed analysis from agentic rewrite, use it
      if (precomputedAnalysis) {
        setAnalysisResults(precomputedAnalysis);
      } else if (!skipAnalysis) {
        // Auto-analyze new version - pass the version object to avoid stale state
        handleAnalyze(promptText, null, newVersion);
      }

    } catch (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  const handleAgenticGenerateEvalPrompt = async () => {
    if (!projectId || !currentVersion) return;

    setIsAgenticGeneratingEval(true);
    setAgenticEvalDetails(null);

    try {
      const response = await fetch(`${API}/step3/agentic-generate-eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: {
            provider: llmProvider,
            api_key: apiKey,
            model_name: llmModel,
            project_name: projectName,
            use_case: useCase,
            requirements: Array.isArray(keyRequirements) ? keyRequirements.join("\n") : keyRequirements,
            initial_prompt: currentVersion.prompt_text
          },
          optimized_prompt: currentVersion.prompt_text,
          use_thinking_model: true
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        // Handle FastAPI validation errors (array) and regular errors (string)
        const errorMessage = Array.isArray(errorData.detail)
          ? errorData.detail.map(e => e.msg || e.message || JSON.stringify(e)).join(', ')
          : (errorData.detail || "Agentic eval generation failed");
        throw new Error(errorMessage);
      }

      const result = await response.json();

      setEvalPrompt(result.eval_prompt);
      setEvalRationale(result.rationale);
      setAgenticEvalDetails(result.agentic_details);

      // Save to project (use PATCH for partial update)
      await fetch(`${API}/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          eval_prompt: result.eval_prompt,
          eval_rationale: result.rationale
        })
      });

      // Also add to eval prompt versions via the generate endpoint
      // The agentic endpoint doesn't create versions, so we manually add one
      await fetch(`${API}/projects/${projectId}/eval-prompt/versions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          eval_prompt_text: result.eval_prompt,
          changes_made: "Agentic generation with failure mode analysis",
          rationale: result.rationale
        })
      });

      // Refresh eval prompt versions
      await fetchEvalPromptVersions();

      const failureModes = result.agentic_details?.failure_modes?.length || 0;
      const confidence = result.agentic_details?.self_test?.confidence_score || 0;

      toast({
        title: "Agentic Eval Prompt Generated",
        description: `Identified ${failureModes} failure modes. Coverage confidence: ${(confidence * 100).toFixed(0)}%`
      });

      setExpandedSections(prev => ({
        ...prev,
        evalPrompt: true
      }));

    } catch (error) {
      toast({
        title: "Agentic Eval Generation Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsAgenticGeneratingEval(false);
    }
  };

  const openModalAndFetchSuggestions = async () => {
    // Open modal first
    setIsEvalAspectsModalOpen(true);

    // Fetch suggestions in background
    if (!projectId) return;

    setIsFetchingSuggestions(true);

    try {
      const response = await fetch(`${API}/projects/${projectId}/eval/suggest-aspects`, {
        method: "GET",
        headers: { "Content-Type": "application/json" }
      });

      if (!response.ok) {
        throw new Error("Failed to fetch suggestions");
      }

      const result = await response.json();

      // Prefill textarea with suggestions (one per line)
      const suggestionsText = result.suggested_aspects.join('\n');
      setEvaluationAspects(suggestionsText);
      setSuggestedDomain(result.domain);
      setSuggestionReasoning(result.reasoning);

      toast({
        title: "✨ Suggestions Generated",
        description: `${result.suggested_aspects.length} aspects suggested for ${result.domain} domain`
      });

    } catch (error) {
      console.error("Error fetching suggestions:", error);
      // Don't show error toast, just leave textarea empty for manual input
    } finally {
      setIsFetchingSuggestions(false);
    }
  };

  const handleMultipleEvalGeneration = async () => {
    if (!projectId || !currentVersion) return;
    if (!evaluationAspects.trim()) {
      toast({
        title: "No Evaluation Aspects",
        description: "Please enter at least one evaluation aspect",
        variant: "destructive"
      });
      return;
    }

    // Close modal and start generation
    setIsEvalAspectsModalOpen(false);
    setIsAgenticGeneratingEval(true);
    setAgenticEvalDetails(null);
    setMultipleEvalPrompts([]);

    try {
      // Split input by lines and filter empty lines
      const aspects = evaluationAspects
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);

      if (aspects.length === 0) {
        throw new Error("No valid evaluation aspects provided");
      }

      toast({
        title: `Generating ${aspects.length} Eval Prompts`,
        description: "Running in parallel with retry logic..."
      });

      // Call BATCH endpoint - generates ALL aspects in PARALLEL
      const response = await fetch(`${API}/projects/${projectId}/eval/generate-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          aspects: aspects,  // Send all aspects at once
          max_retries: 3     // Retry failed aspects
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        const errorMessage = Array.isArray(errorData.detail)
          ? errorData.detail.map(e => e.msg || e.message || JSON.stringify(e)).join(', ')
          : (errorData.detail || "Batch generation failed");
        throw new Error(errorMessage);
      }

      const batchResult = await response.json();

      // Show summary toast
      toast({
        title: `✅ Generation Complete`,
        description: `${batchResult.successful} successful, ${batchResult.failed} failed. Avg: ${(batchResult.average_duration_ms / 1000).toFixed(1)}s each`
      });

      // Convert batch results to UI format (include enhanced fields)
      const generatedPrompts = batchResult.results.map(result => ({
        aspect: result.aspect,
        evalPrompt: result.eval_prompt,
        rationale: result.executive_summary,
        metaQuality: result.meta_quality,
        passesGate: result.passes_gate,
        wasRefined: result.was_refined,
        auditScores: result.audit_scores,
        logicGaps: result.logic_gaps,
        refinementRoadmap: result.refinement_roadmap,
        error: result.error,
        success: result.success,
        attempts: result.attempts,
        durationMs: result.duration_ms,
        // Enhanced best practices fields
        qualityAnalysis: result.quality_analysis,
        calibrationExamples: result.calibration_examples,
        rubricLevels: result.rubric_levels,
        evaluationPurpose: result.evaluation_purpose,
        aiSystemContext: result.ai_system_context
      }));

      setMultipleEvalPrompts(generatedPrompts);

      // NOTE: Multi-aspect prompts are NOT saved as versions
      // These are individual aspect-specific evaluation prompts, not iterations
      // Users can copy any prompt they want to use, but these don't pollute version history

      // Optional: If user wants to save a specific prompt as a version,
      // they can do so manually using the "Save as Version" button (future feature)

      toast({
        title: "Multiple Eval Prompts Generated",
        description: `Successfully generated ${generatedPrompts.length} evaluation prompt(s)`
      });

      setExpandedSections(prev => ({
        ...prev,
        evalPrompt: true
      }));

      // Clear the input
      setEvaluationAspects("");

    } catch (error) {
      toast({
        title: "Multi-Eval Generation Error",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsAgenticGeneratingEval(false);
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

        // Refresh eval prompt versions
        await fetchEvalPromptVersions();

        toast({
          title: "Eval Prompt Refined",
          description: `Applied ${result.changes_made?.length || 0} changes. Version ${result.version || 'new'} created.`
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

  // Test the eval prompt with a sample input/output
  const handleTestEvalPrompt = async () => {
    if (!projectId || !evalPrompt.trim()) {
      toast({
        title: "Eval Prompt Required",
        description: "Please generate an eval prompt first",
        variant: "destructive"
      });
      return;
    }

    if (!evalTestInput.trim() || !evalTestOutput.trim()) {
      toast({
        title: "Sample Input/Output Required",
        description: "Please provide both a sample input and sample output to test",
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

    setIsTestingEval(true);
    setEvalTestResult(null);

    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          eval_prompt: evalPrompt,
          sample_input: evalTestInput,
          sample_output: evalTestOutput,
          expected_score: evalTestExpectedScore ? parseInt(evalTestExpectedScore) : null
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to test eval prompt");
      }

      const result = await response.json();
      setEvalTestResult(result);

      if (result.success) {
        toast({
          title: "Eval Prompt Test Complete",
          description: `Score: ${result.score}/5 - ${result.parsing_status === "success" ? "Parsing successful" : "Partial parsing"}`
        });
      } else {
        toast({
          title: "Eval Prompt Issues Detected",
          description: result.error || "Could not parse score from output",
          variant: "destructive"
        });
      }
    } catch (error) {
      toast({
        title: "Test Failed",
        description: error.message,
        variant: "destructive"
      });
      setEvalTestResult({
        success: false,
        error: error.message,
        parsing_status: "failed"
      });
    } finally {
      setIsTestingEval(false);
    }
  };

  const handleGenerateDataset = async (overrideSampleCount = null) => {
    if (!projectId) return;

    // Use override if provided, otherwise use state
    const countToGenerate = overrideSampleCount ?? sampleCount;

    setIsGeneratingDataset(true);
    setDatasetProgress({ progress: 0, batch: 0, total_batches: 0, status: 'starting' });

    try {
      // Use regular endpoint for dataset generation
      // Include current version number so test cases are relevant to selected prompt
      const response = await fetch(`${API}/projects/${projectId}/dataset/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_count: countToGenerate,
          version: currentVersion?.version
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to generate dataset");
      }

      // Parse JSON response
      const result = await response.json();
      setDataset(result);
      setIsCurrentSessionDataset(true);
      setDatasetProgress({ progress: 100, batch: 0, total_batches: 0, status: 'completed' });

      toast({
        title: "Dataset Generated",
        description: `${result.sample_count || result.count} test cases created`
      });

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

  // Handle CSV file upload for custom dataset
  const handleUploadDataset = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !projectId) return;

    // Validate file type
    if (!file.name.endsWith('.csv')) {
      toast({
        title: "Invalid File",
        description: "Please upload a CSV file",
        variant: "destructive"
      });
      return;
    }

    setIsUploadingDataset(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('replace_existing', 'true');

      const response = await fetch(`${API}/projects/${projectId}/dataset/upload`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Failed to upload dataset");
      }

      // Update local state with the uploaded dataset
      setDataset(result.dataset);
      setIsCurrentSessionDataset(true);

      toast({
        title: "Upload Successful",
        description: `${result.dataset.count} test cases imported from ${file.name}`,
      });

      // Show detected columns info
      if (result.columns_detected) {
        console.log("Detected columns:", result.columns_detected);
      }

    } catch (error) {
      console.error("Upload error:", error);
      toast({
        title: "Upload Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsUploadingDataset(false);
      // Reset file input so same file can be uploaded again
      if (csvUploadRef.current) {
        csvUploadRef.current.value = '';
      }
    }
  };

  // Fetch version diff when comparing exactly 2 versions
  const fetchVersionDiff = async (versionA, versionB) => {
    if (!projectId) return;
    setIsLoadingDiff(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/versions/diff`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version_a: versionA, version_b: versionB })
      });
      if (!response.ok) throw new Error("Failed to fetch diff");
      const data = await response.json();
      setVersionDiffData(data);
    } catch (error) {
      console.error("Error fetching diff:", error);
      setVersionDiffData(null);
    } finally {
      setIsLoadingDiff(false);
    }
  };

  // Fetch eval prompt versions
  const fetchEvalPromptVersions = async () => {
    if (!projectId) return;
    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/versions`);
      if (!response.ok) return;
      const data = await response.json();
      setEvalPromptVersions(data.versions || []);
      // Set current version
      const current = data.versions?.find(v => v.is_current);
      if (current) {
        setCurrentEvalVersion(current);
      }
    } catch (error) {
      console.error("Error fetching eval prompt versions:", error);
    }
  };

  // Fetch eval version diff
  const fetchEvalVersionDiff = async (versionA, versionB) => {
    if (!projectId) return;
    setIsLoadingEvalDiff(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/versions/diff`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version_a: versionA, version_b: versionB })
      });
      if (!response.ok) throw new Error("Failed to fetch eval diff");
      const data = await response.json();
      setEvalVersionDiffData(data);
    } catch (error) {
      console.error("Error fetching eval diff:", error);
      setEvalVersionDiffData(null);
    } finally {
      setIsLoadingEvalDiff(false);
    }
  };

  // Restore eval prompt version
  const handleRestoreEvalVersion = async (versionNumber) => {
    if (!projectId) return;
    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/versions/${versionNumber}/restore`, {
        method: "PUT"
      });
      if (!response.ok) throw new Error("Failed to restore version");
      const data = await response.json();
      setEvalPrompt(data.eval_prompt);
      setEvalRationale(data.rationale);
      await fetchEvalPromptVersions();
      toast({
        title: "Version Restored",
        description: `Restored eval prompt version ${versionNumber}`
      });
    } catch (error) {
      toast({
        title: "Restore Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  // Delete eval prompt version
  const handleDeleteEvalVersion = async (versionNumber) => {
    if (!projectId) return;
    if (evalPromptVersions.length <= 1) {
      toast({
        title: "Cannot Delete",
        description: "Cannot delete the only version",
        variant: "destructive"
      });
      return;
    }
    try {
      const response = await fetch(`${API}/projects/${projectId}/eval-prompt/versions/${versionNumber}`, {
        method: "DELETE"
      });
      if (!response.ok) throw new Error("Failed to delete version");
      await fetchEvalPromptVersions();
      toast({
        title: "Version Deleted",
        description: `Deleted eval prompt version ${versionNumber}`
      });
    } catch (error) {
      toast({
        title: "Delete Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  // Check for regression when version changes
  const checkRegression = async (versionNum = null) => {
    if (!projectId) return;
    setIsCheckingRegression(true);
    try {
      const url = versionNum
        ? `${API}/projects/${projectId}/versions/regression-check?version=${versionNum}`
        : `${API}/projects/${projectId}/versions/regression-check`;
      const response = await fetch(url);
      if (!response.ok) throw new Error("Failed to check regression");
      const data = await response.json();
      setRegressionAlert(data);

      // Show toast if regression detected
      if (data.has_regression) {
        toast({
          title: "Regression Detected",
          description: data.recommendation,
          variant: "destructive",
          duration: 10000
        });
      }
    } catch (error) {
      console.error("Error checking regression:", error);
      setRegressionAlert(null);
    } finally {
      setIsCheckingRegression(false);
    }
  };

  // Check regression after analysis completes
  useEffect(() => {
    if (analysisResults && versionHistory.length >= 2 && currentVersion) {
      checkRegression(currentVersion.version);
    }
  }, [analysisResults]);

  // ============= Regenerate Handlers =============

  // Regenerate analysis with existing prompt - sends to LLM for fresh analysis
  const handleRegenerateAnalysis = async () => {
    if (!projectId || !currentVersion?.prompt_text) {
      toast({
        title: "No Prompt",
        description: "Please create a prompt first",
        variant: "destructive"
      });
      return;
    }

    setIsRegeneratingAnalysis(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: currentVersion.prompt_text,
          regenerate: true // Signal to backend this is a regeneration
        })
      });

      if (!response.ok) throw new Error("Failed to regenerate analysis");

      const result = await response.json();
      setAnalysisResults(result);

      toast({
        title: "Analysis Regenerated",
        description: "Fresh analysis has been generated"
      });
    } catch (error) {
      toast({
        title: "Regeneration Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRegeneratingAnalysis(false);
    }
  };

  // Regenerate/improve the system prompt using existing prompt and analysis
  const handleRegeneratePrompt = async () => {
    if (!projectId || !currentVersion?.prompt_text) {
      toast({
        title: "No Prompt",
        description: "Please create a prompt first",
        variant: "destructive"
      });
      return;
    }

    setIsRegeneratingPrompt(true);
    try {
      const response = await fetch(`${API}/projects/${projectId}/rewrite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: currentVersion.prompt_text,
          feedback: "Improve this prompt based on the analysis. Make it more effective, clearer, and better structured.",
          focus_areas: analysisResults?.suggestions?.map(s => s.suggestion) || []
        })
      });

      if (!response.ok) throw new Error("Failed to regenerate prompt");

      const result = await response.json();

      // Add as new version
      const addVersionResponse = await fetch(`${API}/projects/${projectId}/versions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: result.rewritten_prompt,
          changes_made: "Regenerated with AI improvements"
        })
      });

      if (!addVersionResponse.ok) throw new Error("Failed to save regenerated version");

      const newVersion = await addVersionResponse.json();
      setVersionHistory(prev => [...prev, newVersion]);
      setCurrentVersion(newVersion);
      setPromptChanges(result.changes_made || ["AI-improved version"]);

      toast({
        title: "Prompt Regenerated",
        description: `Version ${newVersion.version} created with improvements`
      });
    } catch (error) {
      toast({
        title: "Regeneration Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRegeneratingPrompt(false);
    }
  };

  // Regenerate eval prompt using agentic flow for smart, context-aware generation
  const handleRegenerateEvalPrompt = async () => {
    if (!projectId) {
      toast({
        title: "No Project",
        description: "Please create a project first",
        variant: "destructive"
      });
      return;
    }

    // Check if we have LLM settings configured
    if (!settingsLoaded || !apiKey) {
      toast({
        title: "Settings Required",
        description: "Please configure your LLM settings first (click the gear icon)",
        variant: "destructive"
      });
      return;
    }

    setIsRegeneratingEval(true);
    setAgenticEvalDetails(null);

    try {
      // Use agentic flow if we have a system prompt (for smart, domain-aware generation)
      // This analyzes the system prompt to generate context-appropriate eval dimensions
      const hasSystemPrompt = currentVersion?.prompt_text && currentVersion.prompt_text.trim().length > 0;

      if (hasSystemPrompt) {
        // Use agentic flow for smart regeneration
        const response = await fetch(`${API}/step3/agentic-generate-eval`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project: {
              provider: llmProvider,
              api_key: apiKey,
              model_name: llmModel,
              project_name: projectName,
              use_case: useCase,
              requirements: Array.isArray(keyRequirements) ? keyRequirements.join("\n") : keyRequirements,
              initial_prompt: currentVersion.prompt_text
            },
            optimized_prompt: currentVersion.prompt_text,
            use_thinking_model: true,
            // Pass existing eval prompt for context-aware improvement
            existing_eval_prompt: evalPrompt,
            existing_rationale: evalRationale
          })
        });

        if (!response.ok) {
          const errorData = await response.json();
          const errorMessage = Array.isArray(errorData.detail)
            ? errorData.detail.map(e => e.msg || e.message || JSON.stringify(e)).join(', ')
            : (errorData.detail || "Agentic regeneration failed");
          throw new Error(errorMessage);
        }

        const result = await response.json();

        setEvalPrompt(result.eval_prompt);
        setEvalRationale(result.rationale);
        setAgenticEvalDetails(result.agentic_details);

        // Save to project
        await fetch(`${API}/projects/${projectId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            eval_prompt: result.eval_prompt,
            eval_rationale: result.rationale
          })
        });

        // Add to eval prompt versions
        await fetch(`${API}/projects/${projectId}/eval-prompt/versions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            eval_prompt_text: result.eval_prompt,
            changes_made: "Smart regeneration with failure mode analysis",
            rationale: result.rationale
          })
        });

        // Refresh eval prompt versions
        await fetchEvalPromptVersions();

        const failureModes = result.agentic_details?.failure_modes?.length || 0;
        const dimensions = result.agentic_details?.eval_dimensions?.length || 0;

        toast({
          title: "Eval Prompt Regenerated (Smart)",
          description: `Generated ${dimensions} domain-specific dimensions based on ${failureModes} identified failure modes.`
        });

      } else {
        // Fallback to simpler regeneration for imported eval prompts without system prompt
        const response = await fetch(`${API}/projects/${projectId}/eval-prompt/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            regenerate: true,
            current_eval_prompt: evalPrompt,
            current_rationale: evalRationale,
            eval_changes: evalChanges
          })
        });

        if (!response.ok) throw new Error("Failed to regenerate eval prompt");

        const result = await response.json();
        setEvalPrompt(result.eval_prompt);
        setEvalRationale(result.rationale);

        if (result.is_regeneration) {
          setEvalChanges(prev => [...prev, "Improved using Anthropic's eval best practices"]);
        }

        await fetchEvalPromptVersions();

        toast({
          title: "Eval Prompt Improved",
          description: `Evaluation prompt improved. Version ${result.version || 'new'} created.`
        });
      }

    } catch (error) {
      toast({
        title: "Regeneration Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsRegeneratingEval(false);
    }
  };

  // Regenerate dataset with same parameters (new version)
  const handleRegenerateDatasetNew = async () => {
    if (!projectId) {
      toast({
        title: "No Project",
        description: "Please create a project first",
        variant: "destructive"
      });
      return;
    }

    const countToGenerate = dataset?.sample_count || sampleCount;

    setIsRegeneratingDataset(true);
    setDatasetProgress({ progress: 0, batch: 0, total_batches: 0, status: 'regenerating' });

    try {
      const response = await fetch(`${API}/projects/${projectId}/dataset/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_count: countToGenerate,
          version: currentVersion?.version,
          regenerate: true // Signal this is a regeneration
        })
      });

      if (!response.ok) throw new Error("Failed to regenerate dataset");

      const result = await response.json();
      setDataset(result);
      setIsCurrentSessionDataset(true);
      setDatasetProgress({ progress: 100, batch: 0, total_batches: 0, status: 'completed' });

      toast({
        title: "Dataset Regenerated",
        description: `${result.sample_count} new test cases created`
      });
    } catch (error) {
      toast({
        title: "Regeneration Failed",
        description: error.message,
        variant: "destructive"
      });
      setDatasetProgress({ progress: 0, batch: 0, total_batches: 0, status: 'error' });
    } finally {
      setIsRegeneratingDataset(false);
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
          // Evaluation model settings
          eval_provider: useSeparateEvalModel ? evalProvider : llmProvider,
          eval_model: useSeparateEvalModel ? evalModel : (llmModel || null),
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

      // Check if test run completed immediately (sync execution)
      if (data.status === "completed") {
        setTestRunStatus({
          status: "completed",
          progress: 100,
          completed_items: data.total_items,
          total_items: data.total_items,
          summary: data.summary
        });

        // Set results directly from response
        setTestRunResults({
          results: data.results,
          summary: data.summary
        });

        loadTestRunHistory();

        toast({
          title: "Test Run Complete",
          description: `${data.summary?.passed || 0} of ${data.summary?.total || 0} tests passed`
        });
      } else {
        // Async execution - start polling
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

        pollTestRunStatus(data.run_id);
      }

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
      // Transform to ensure consistent format with run_id
      const formattedRuns = data.map(run => ({
        ...run,
        run_id: run.run_id || run.id  // Ensure run_id exists
      }));
      setTestRunHistory(formattedRuns);
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

    if (!runId) {
      console.error("Cannot delete: run_id is undefined");
      return;
    }

    if (!window.confirm("Are you sure you want to delete this test run? This action cannot be undone.")) {
      return;
    }

    try {
      const response = await fetch(`${API}/projects/${projectId}/test-runs/${runId}`, {
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
                  {/* Hidden File Input */}
                  <input
                    type="file"
                    accept=".txt"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    onChange={handleFileSelected}
                  />

                  {/* New Project / Import Buttons */}
                  <div className="flex gap-2">
                    <Button
                      onClick={resetToNewProject}
                      className="flex-1 bg-blue-600 hover:bg-blue-700"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Create New Project
                    </Button>
                    <Button
                      onClick={handleImportClick}
                      variant="outline"
                      className="flex-1 border-blue-600 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                    >
                      <Upload className="w-4 h-4 mr-2" />
                      Import Eval Prompt
                    </Button>
                  </div>

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
                              <div className="flex items-center gap-2">
                                <h4 className="font-medium text-slate-900 dark:text-slate-100">{project.project_name || project.name}</h4>
                                {project.project_type === 'eval' && (
                                  <span className="px-2 py-0.5 text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded font-medium">
                                    Eval
                                  </span>
                                )}
                              </div>
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
                                onClick={(e) => handleDeleteProject(project.id, project.project_name || project.name, e)}
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

            {/* Import Eval Prompt Dialog */}
            <Dialog open={showImportDialog} onOpenChange={(open) => {
              setShowImportDialog(open);
              if (!open) {
                setImportedEvalPrompt("");
                setImportProjectName("");
                setImportDescription("");
              }
            }}>
              <DialogContent className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-900 dark:text-white max-w-lg">
                <DialogHeader>
                  <DialogTitle className="text-xl text-slate-900 dark:text-white">Import Eval Prompt</DialogTitle>
                  <DialogDescription className="text-slate-600 dark:text-slate-400">
                    Create a new project with your imported evaluation prompt
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label className="text-slate-700 dark:text-slate-300">Project Name *</Label>
                    <Input
                      value={importProjectName}
                      onChange={(e) => setImportProjectName(e.target.value)}
                      placeholder="Enter a name for this project"
                      className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700 dark:text-slate-300">What does this eval prompt evaluate? *</Label>
                    <Textarea
                      value={importDescription}
                      onChange={(e) => setImportDescription(e.target.value)}
                      placeholder="Brief description of what this evaluation prompt is designed to assess..."
                      rows={3}
                      className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700 dark:text-slate-300">Imported Eval Prompt Preview</Label>
                    <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 max-h-32 overflow-y-auto">
                      <pre className="text-xs text-slate-600 dark:text-slate-400 whitespace-pre-wrap">
                        {importedEvalPrompt.slice(0, 500)}{importedEvalPrompt.length > 500 ? '...' : ''}
                      </pre>
                    </div>
                  </div>
                </div>
                <DialogFooter className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setShowImportDialog(false)}
                    className="border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleCreateImportedProject}
                    disabled={isImporting || !importProjectName.trim() || !importDescription.trim()}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    {isImporting ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <Plus className="w-4 h-4 mr-2" />
                        Create Project
                      </>
                    )}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            {/* Settings Button */}
            <Button
              variant="ghost"
              size="icon"
              className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-700"
              title="Settings"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings className="h-5 w-5" />
            </Button>
            <SettingsModal
              open={settingsOpen}
              onOpenChange={setSettingsOpen}
              settings={{
                llmProvider,
                llmModel,
                apiKey
              }}
              onSettingsChange={(newSettings) => {
                if (newSettings.llmProvider !== undefined) setLlmProvider(newSettings.llmProvider);
                if (newSettings.llmModel !== undefined) setLlmModel(newSettings.llmModel);
                if (newSettings.apiKey !== undefined) setApiKey(newSettings.apiKey);
              }}
              onSave={(savedSettings) => {
                // Sync state after save from modal
                if (savedSettings) {
                  setLlmProvider(savedSettings.llm_provider || llmProvider);
                  setLlmModel(savedSettings.model_name || llmModel);
                  setApiKey(savedSettings.api_key || apiKey);
                  setSettingsLoaded(true);
                  // Also save to localStorage
                  localStorage.setItem('athena_llm_settings', JSON.stringify(savedSettings));
                }
              }}
            />
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
                  Cannot connect to the backend server at <code className="bg-slate-800 px-1 py-0.5 rounded">{BASE_URL}</code>
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
                  <Label htmlFor="projectName" className="text-slate-700 dark:text-slate-300 font-medium">Project Name *</Label>
                  <Input
                    id="projectName"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    placeholder="e.g., Customer Support Bot"
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100"
                  />
                </div>

                <div>
                  <Label htmlFor="initialPrompt" className="text-slate-700 dark:text-slate-300 font-medium">Initial System Prompt *</Label>
                  <Textarea
                    id="initialPrompt"
                    value={initialPrompt}
                    onChange={(e) => handleInitialPromptChange(e.target.value)}
                    placeholder="Enter your initial system prompt here..."
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[200px] font-mono text-sm"
                  />
                  {isExtractingFromPrompt && (
                    <div className="flex items-center gap-2 mt-2 p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-700 rounded-lg">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-500"></div>
                      <span className="text-sm text-purple-600 dark:text-purple-400">
                        Analyzing prompt to extract use case and requirements...
                      </span>
                    </div>
                  )}
                  {!isExtractingFromPrompt && initialPrompt.trim().length > 0 && !useCase.trim() && !keyRequirements.trim() && initialPrompt.trim().length < 50 && (
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Keep typing... Use case and requirements will be auto-extracted once prompt is longer
                    </p>
                  )}
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <Label htmlFor="useCase" className="text-slate-700 dark:text-slate-300 font-medium">Use Case *</Label>
                    {useCase && initialPrompt.trim().length >= 50 && (
                      <span className="text-xs text-purple-500 dark:text-purple-400 flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        Auto-filled - review and edit as needed
                      </span>
                    )}
                  </div>
                  <Textarea
                    id="useCase"
                    value={useCase}
                    onChange={(e) => setUseCase(e.target.value)}
                    placeholder="Describe what this system prompt is for..."
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[80px]"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <Label htmlFor="requirements" className="text-slate-700 dark:text-slate-300 font-medium">Key Requirements * (one per line or comma-separated)</Label>
                    {keyRequirements && initialPrompt.trim().length >= 50 && (
                      <span className="text-xs text-purple-500 dark:text-purple-400 flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        Auto-filled
                      </span>
                    )}
                  </div>
                  <Textarea
                    id="requirements"
                    value={keyRequirements}
                    onChange={(e) => setKeyRequirements(e.target.value)}
                    placeholder="e.g.,&#10;- Handle customer queries professionally&#10;- Provide accurate product information&#10;- Escalate complex issues to human agents"
                    className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[120px]"
                  />
                </div>

                {/* Enhanced Context (Optional) - PRD Import + Specific Fields */}
                <div className="border-t border-slate-300 dark:border-slate-700 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowEnhancedContext(!showEnhancedContext)}
                    className="flex items-center justify-between w-full text-left p-3 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg hover:from-blue-100 hover:to-purple-100 dark:hover:from-blue-900/30 dark:hover:to-purple-900/30 transition-colors border border-blue-200 dark:border-blue-800"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        📄 Enhanced Context (Optional)
                      </span>
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {showEnhancedContext ? 'Click to collapse' : 'Add PRD or specific context for better generation'}
                      </span>
                    </div>
                    <span className="text-slate-500 dark:text-slate-400">
                      {showEnhancedContext ? '▼' : '▶'}
                    </span>
                  </button>
                  
                  {showEnhancedContext && (
                    <div className="mt-4 space-y-4 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
                      {/* PRD Document Import */}
                      <div>
                        <Label htmlFor="prdDocument" className="text-slate-700 dark:text-slate-300 text-sm font-semibold">
                          📋 PRD / Requirements Document
                        </Label>
                        <Textarea
                          id="prdDocument"
                          value={prdDocument}
                          onChange={(e) => setPrdDocument(e.target.value)}
                          placeholder="Paste your Product Requirements Document here...&#10;&#10;The system will automatically extract:&#10;- User personas and target audience&#10;- Success criteria and metrics&#10;- Edge cases and constraints&#10;- Must-do and forbidden behaviors&#10;&#10;Supports: Plain text, Markdown"
                          className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[150px] text-sm mt-2 font-mono"
                        />
                        <div className="flex items-start gap-2 mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                          <span className="text-blue-600 dark:text-blue-400 text-xs">💡</span>
                          <p className="text-xs text-blue-700 dark:text-blue-300">
                            AI will analyze your PRD to extract must-do behaviors, constraints, success criteria, and edge cases automatically.
                          </p>
                        </div>
                      </div>

                      {/* Specific Context Fields */}
                      <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
                        <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-3">SPECIFIC CONTEXT (Optional)</p>
                        
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div>
                            <Label htmlFor="tone" className="text-slate-700 dark:text-slate-300 text-sm">Required Tone</Label>
                            <input
                              id="tone"
                              type="text"
                              value={tone}
                              onChange={(e) => setTone(e.target.value)}
                              placeholder="e.g., Professional, Empathetic"
                              className="w-full p-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md text-sm mt-1"
                            />
                          </div>

                          <div>
                            <Label htmlFor="outputFormat" className="text-slate-700 dark:text-slate-300 text-sm">Output Format</Label>
                            <input
                              id="outputFormat"
                              type="text"
                              value={outputFormat}
                              onChange={(e) => setOutputFormat(e.target.value)}
                              placeholder="e.g., JSON, Structured text"
                              className="w-full p-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md text-sm mt-1"
                            />
                          </div>
                        </div>

                        <div className="mb-4">
                          <Label htmlFor="successCriteria" className="text-slate-700 dark:text-slate-300 text-sm">Success Criteria (comma-separated)</Label>
                          <Textarea
                            id="successCriteria"
                            value={successCriteria}
                            onChange={(e) => setSuccessCriteria(e.target.value)}
                            placeholder="e.g., Response time < 3s, Customer satisfaction > 4.5, Resolution rate > 80%"
                            className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[60px] text-sm mt-1"
                          />
                        </div>

                        <div>
                          <Label htmlFor="edgeCases" className="text-slate-700 dark:text-slate-300 text-sm">Known Edge Cases (comma-separated)</Label>
                          <Textarea
                            id="edgeCases"
                            value={edgeCases}
                            onChange={(e) => setEdgeCases(e.target.value)}
                            placeholder="e.g., Angry customers, Technical jargon, International customers"
                            className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[60px] text-sm mt-1"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div>
                  <Label htmlFor="provider" className="text-slate-700 dark:text-slate-300 font-medium">Target LLM Provider *</Label>
                  <Select value={targetProvider} onValueChange={setTargetProvider}>
                    <SelectTrigger className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                      <SelectItem value="openai" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">OpenAI (GPT)</SelectItem>
                      <SelectItem value="claude" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">Anthropic (Claude)</SelectItem>
                      <SelectItem value="gemini" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">Google (Gemini)</SelectItem>
                      <SelectItem value="multi" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">Multi-provider</SelectItem>
                    </SelectContent>
                  </Select>
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

                      {/* Dual Score Breakdown - Pattern vs LLM Analysis */}
                      {analysisResults.enhanced_analysis && analysisResults.enhanced_analysis.llm_score > 0 && (
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <div className="text-xs font-medium text-slate-400 mb-3">Score Analysis Breakdown</div>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="text-center p-3 bg-slate-800 rounded border border-slate-600">
                              <div className="text-xs text-slate-400">Pattern Analysis</div>
                              <div className={`text-2xl font-bold ${
                                analysisResults.enhanced_analysis.programmatic_score >= 8 ? 'text-green-400' :
                                analysisResults.enhanced_analysis.programmatic_score >= 6 ? 'text-yellow-400' : 'text-red-400'
                              }`}>
                                {analysisResults.enhanced_analysis.programmatic_score?.toFixed(1)}/10
                              </div>
                              <div className="text-xs text-slate-500">Rule-based (30% weight)</div>
                            </div>
                            <div className="text-center p-3 bg-slate-800 rounded border border-slate-600">
                              <div className="text-xs text-slate-400">LLM Analysis</div>
                              <div className={`text-2xl font-bold ${
                                analysisResults.enhanced_analysis.llm_score >= 8 ? 'text-green-400' :
                                analysisResults.enhanced_analysis.llm_score >= 6 ? 'text-yellow-400' : 'text-red-400'
                              }`}>
                                {analysisResults.enhanced_analysis.llm_score?.toFixed(1)}/10
                              </div>
                              <div className="text-xs text-slate-500">Semantic (70% weight)</div>
                            </div>
                          </div>
                          {Math.abs(analysisResults.enhanced_analysis.programmatic_score - analysisResults.enhanced_analysis.llm_score) > 1.5 && (
                            <div className="mt-3 text-xs text-yellow-400 flex items-center gap-1">
                              <AlertTriangle className="h-3 w-3" />
                              Score discrepancy: Pattern analysis found issues that LLM considers acceptable. Review suggestions carefully.
                            </div>
                          )}
                        </div>
                      )}

                      {analysisResults.requirements_gaps.length > 0 && (
                        <div className="bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-lg p-4">
                          <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">Missing Requirements:</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-red-700 dark:text-red-200">
                            {analysisResults.requirements_gaps.map((gap, idx) => (
                              <li key={idx}>{gap}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {analysisResults.suggestions.length > 0 && (
                        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg p-4">
                          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Top Suggestions:</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-blue-700 dark:text-blue-200">
                            {analysisResults.suggestions.slice(0, 5).map((sug, idx) => (
                              <li key={idx}>
                                <span className="font-medium text-blue-800 dark:text-blue-300">[{sug.priority}]</span> {sug.suggestion}
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
                    <Button
                      onClick={handleAgenticRewrite}
                      disabled={isAgenticRewriting}
                      className="bg-purple-600 hover:bg-purple-700 text-white"
                      title="Multi-step agentic rewrite with thinking model"
                    >
                      {isAgenticRewriting ? (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          Rewriting...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4 mr-2" />
                          AI Rewrite
                        </>
                      )}
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

                  {/* Regression Alert Banner */}
                  {regressionAlert && regressionAlert.has_regression && (
                    <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-lg">
                      <div className="flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <h4 className="font-semibold text-red-700 dark:text-red-400">Regression Detected</h4>
                          <p className="text-sm text-red-600 dark:text-red-300 mt-1">
                            Version {regressionAlert.current_version} scored {regressionAlert.current_score} vs Version {regressionAlert.previous_version}'s {regressionAlert.previous_score} ({regressionAlert.score_change > 0 ? '+' : ''}{regressionAlert.score_change} points)
                          </p>
                          <p className="text-xs text-red-500 dark:text-red-400 mt-2">{regressionAlert.recommendation}</p>
                          <div className="flex gap-2 mt-3">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => {
                                // Load the previous (better) version
                                const prevVersion = versionHistory.find(v => v.version === regressionAlert.previous_version);
                                if (prevVersion) {
                                  setCurrentVersion(prevVersion);
                                  toast({ title: "Reverted", description: `Switched to Version ${regressionAlert.previous_version}` });
                                }
                              }}
                              className="border-red-500 text-red-600 hover:bg-red-100 dark:hover:bg-red-900/30"
                            >
                              Revert to Version {regressionAlert.previous_version}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setRegressionAlert(null)}
                              className="text-slate-600 dark:text-slate-400"
                            >
                              Dismiss
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Improvement Alert Banner */}
                  {regressionAlert && !regressionAlert.has_regression && regressionAlert.score_change > 5 && (
                    <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg">
                      <div className="flex items-start gap-3">
                        <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <h4 className="font-semibold text-green-700 dark:text-green-400">Improvement Detected</h4>
                          <p className="text-sm text-green-600 dark:text-green-300 mt-1">
                            Version {regressionAlert.current_version} scored {regressionAlert.current_score} vs Version {regressionAlert.previous_version}'s {regressionAlert.previous_score} (+{regressionAlert.score_change} points)
                          </p>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setRegressionAlert(null)}
                            className="text-slate-600 dark:text-slate-400 mt-2"
                          >
                            Dismiss
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Version History */}
                  {versionHistory.length > 1 && (
                    <div className="mt-6">
                      <div className="flex justify-between items-center mb-2">
                        <div>
                          <h3 className="font-semibold text-slate-900 dark:text-slate-100">Version History</h3>
                          <p className="text-xs text-slate-500 dark:text-slate-400">Click on a version to load it, or select versions to compare</p>
                        </div>
                        {selectedVersionsForCompare.length >= 2 && (
                          <Button
                            size="sm"
                            onClick={() => {
                              setVersionCompareModalOpen(true);
                              // Fetch diff if exactly 2 versions selected
                              if (selectedVersionsForCompare.length === 2) {
                                const sorted = [...selectedVersionsForCompare].sort((a, b) => a - b);
                                fetchVersionDiff(sorted[0], sorted[1]);
                              }
                            }}
                            className="bg-purple-600 hover:bg-purple-700 text-white"
                          >
                            <ArrowUpDown className="w-4 h-4 mr-1" />
                            Compare ({selectedVersionsForCompare.length})
                          </Button>
                        )}
                      </div>
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
                            className={`p-3 rounded-lg border transition-all cursor-pointer ${
                              v.version === currentVersion?.version
                                ? 'bg-blue-100 dark:bg-blue-900/20 border-blue-600 dark:border-blue-700'
                                : 'bg-slate-50 dark:bg-slate-800 border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-slate-100 dark:hover:bg-slate-700'
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <div className="flex items-center gap-3">
                                {/* Checkbox for comparison */}
                                <input
                                  type="checkbox"
                                  checked={selectedVersionsForCompare.includes(v.version)}
                                  onChange={(e) => {
                                    e.stopPropagation();
                                    if (e.target.checked) {
                                      setSelectedVersionsForCompare(prev => [...prev, v.version]);
                                    } else {
                                      setSelectedVersionsForCompare(prev => prev.filter(ver => ver !== v.version));
                                    }
                                  }}
                                  onClick={(e) => e.stopPropagation()}
                                  className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 text-purple-600 focus:ring-purple-500"
                                />
                                <div className="flex items-center gap-2 flex-1">
                                  <span className="font-medium text-slate-900 dark:text-white">Version {v.version}</span>
                                  {v.version === currentVersion?.version && (
                                    <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded">Current</span>
                                  )}
                                  {v.is_final && <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded">Final</span>}
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                {v.evaluation && (
                                  <span className={`text-sm font-semibold px-2 py-0.5 rounded ${
                                    (v.evaluation.overall_score || ((v.evaluation.requirements_alignment + v.evaluation.best_practices_score) / 2)) >= 80
                                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                      : (v.evaluation.overall_score || ((v.evaluation.requirements_alignment + v.evaluation.best_practices_score) / 2)) >= 60
                                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                  }`}>
                                    {(v.evaluation.overall_score || ((v.evaluation.requirements_alignment + v.evaluation.best_practices_score) / 2)).toFixed(1)}
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
                  {/* Multiple Eval Prompts Display */}
                  {multipleEvalPrompts.length > 0 && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-slate-700 dark:text-slate-300">
                          Generated Evaluation Prompts ({multipleEvalPrompts.length})
                        </h3>
                        <Button
                          onClick={() => {
                            setMultipleEvalPrompts([]);
                            setEvaluationAspects("");
                          }}
                          variant="outline"
                          size="sm"
                        >
                          <X className="w-4 h-4 mr-1" />
                          Clear All
                        </Button>
                      </div>

                      {multipleEvalPrompts.map((promptData, index) => (
                        <Collapsible
                          key={index}
                          open={expandedEvalPrompts.has(index)}
                          onOpenChange={(isOpen) => {
                            const newSet = new Set(expandedEvalPrompts);
                            if (isOpen) {
                              newSet.add(index);
                            } else {
                              newSet.delete(index);
                            }
                            setExpandedEvalPrompts(newSet);
                          }}
                          className="border border-slate-300 dark:border-slate-700 rounded-lg overflow-hidden"
                        >
                          {/* Header - Clickable to toggle */}
                          <CollapsibleTrigger className="w-full">
                            <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border-b border-slate-300 dark:border-slate-700 px-4 py-3 hover:from-purple-100 hover:to-blue-100 dark:hover:from-purple-900/30 dark:hover:to-blue-900/30 transition-colors cursor-pointer">
                              <div className="flex items-start justify-between">
                                <div className="flex items-center gap-2 flex-1">
                                  {expandedEvalPrompts.has(index) ? (
                                    <ChevronDown className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                                  ) : (
                                    <ChevronRight className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                                  )}
                                  <div className="flex-1 text-left">
                                    <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-1">
                                      {index + 1}. {promptData.aspect}
                                    </h4>
                                <div className="flex items-center gap-3 text-xs">
                                  {promptData.metaQuality && (
                                    <span className={`px-2 py-1 rounded ${
                                      promptData.passesGate
                                        ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                                        : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
                                    }`}>
                                      Quality: {promptData.metaQuality.toFixed(1)}/10
                                    </span>
                                  )}
                                  {promptData.wasRefined && (
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                                      ✨ Auto-Refined
                                    </span>
                                  )}
                                  {promptData.passesGate && (
                                    <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded">
                                      ✓ Passes Quality Gate
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        </CollapsibleTrigger>

                          {/* Collapsible Content */}
                          <CollapsibleContent>
                            {/* Quality Analysis - Best Practices Implementation */}
                          {promptData.qualityAnalysis && (
                            <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-green-700 dark:text-green-400 mb-2">
                                ✓ Industry Best Practices Implementation
                              </h5>
                              <div className="grid grid-cols-3 gap-3">
                                <div className={`flex items-center gap-2 text-xs ${
                                  promptData.qualityAnalysis.has_chain_of_thought
                                    ? 'text-green-700 dark:text-green-300'
                                    : 'text-slate-500 dark:text-slate-400'
                                }`}>
                                  <span>{promptData.qualityAnalysis.has_chain_of_thought ? '✓' : '○'}</span>
                                  <span>Chain-of-Thought</span>
                                </div>
                                <div className={`flex items-center gap-2 text-xs ${
                                  promptData.qualityAnalysis.has_0_10_scale
                                    ? 'text-green-700 dark:text-green-300'
                                    : 'text-slate-500 dark:text-slate-400'
                                }`}>
                                  <span>{promptData.qualityAnalysis.has_0_10_scale ? '✓' : '○'}</span>
                                  <span>0-10 Scale</span>
                                </div>
                                <div className={`flex items-center gap-2 text-xs ${
                                  promptData.qualityAnalysis.has_xml_format
                                    ? 'text-green-700 dark:text-green-300'
                                    : 'text-slate-500 dark:text-slate-400'
                                }`}>
                                  <span>{promptData.qualityAnalysis.has_xml_format ? '✓' : '○'}</span>
                                  <span>Structured Output</span>
                                </div>
                                <div className={`flex items-center gap-2 text-xs ${
                                  promptData.qualityAnalysis.num_examples >= 3
                                    ? 'text-green-700 dark:text-green-300'
                                    : 'text-slate-500 dark:text-slate-400'
                                }`}>
                                  <span>{promptData.qualityAnalysis.num_examples >= 3 ? '✓' : '○'}</span>
                                  <span>{promptData.qualityAnalysis.num_examples} Examples</span>
                                </div>
                                <div className={`flex items-center gap-2 text-xs ${
                                  promptData.qualityAnalysis.has_behavioral_anchors
                                    ? 'text-green-700 dark:text-green-300'
                                    : 'text-slate-500 dark:text-slate-400'
                                }`}>
                                  <span>{promptData.qualityAnalysis.has_behavioral_anchors ? '✓' : '○'}</span>
                                  <span>Behavioral Rubric</span>
                                </div>
                                <div className={`flex items-center gap-2 text-xs ${
                                  promptData.qualityAnalysis.has_uncertainty_handling
                                    ? 'text-green-700 dark:text-green-300'
                                    : 'text-slate-500 dark:text-slate-400'
                                }`}>
                                  <span>{promptData.qualityAnalysis.has_uncertainty_handling ? '✓' : '○'}</span>
                                  <span>Uncertainty Handling</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* AI System Context */}
                          {promptData.aiSystemContext && (
                            <div className="bg-sky-50 dark:bg-sky-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-sky-600 dark:text-sky-400 mb-2">
                                AI System Context:
                              </h5>
                              {promptData.aiSystemContext.system_summary && (
                                <div className="mb-2">
                                  <span className="text-xs font-medium text-sky-700 dark:text-sky-300">What This System Does: </span>
                                  <span className="text-xs text-slate-700 dark:text-slate-300">
                                    {promptData.aiSystemContext.system_summary}
                                  </span>
                                </div>
                              )}
                              {promptData.aiSystemContext.use_case && (
                                <div className="mb-2">
                                  <span className="text-xs font-medium text-sky-700 dark:text-sky-300">Use Case: </span>
                                  <span className="text-xs text-slate-700 dark:text-slate-300">
                                    {promptData.aiSystemContext.use_case}
                                  </span>
                                </div>
                              )}
                              {promptData.aiSystemContext.system_prompt && (
                                <details className="mt-2">
                                  <summary className="text-xs font-medium text-sky-700 dark:text-sky-300 cursor-pointer hover:text-sky-800 dark:hover:text-sky-200">
                                    View Full System Prompt
                                  </summary>
                                  <pre className="mt-2 text-xs bg-white dark:bg-slate-900 p-2 rounded border border-sky-200 dark:border-sky-800 overflow-x-auto whitespace-pre-wrap max-h-32 overflow-y-auto">
{promptData.aiSystemContext.system_prompt}
                                  </pre>
                                </details>
                              )}
                            </div>
                          )}

                          {/* Evaluation Purpose */}
                          {promptData.evaluationPurpose && (
                            <div className="bg-indigo-50 dark:bg-indigo-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 mb-1">Purpose:</h5>
                              <p className="text-xs text-slate-700 dark:text-slate-300">
                                {promptData.evaluationPurpose}
                              </p>
                            </div>
                          )}

                          {/* Rubric Levels */}
                          {promptData.rubricLevels && promptData.rubricLevels.length > 0 && (
                            <div className="bg-violet-50 dark:bg-violet-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-violet-600 dark:text-violet-400 mb-2">
                                Scoring Rubric (0-10 Scale):
                              </h5>
                              <div className="space-y-2">
                                {promptData.rubricLevels.map((level, i) => (
                                  <div key={i} className="text-xs">
                                    <div className="font-semibold text-violet-700 dark:text-violet-300">
                                      {level.level}
                                    </div>
                                    <div className="text-slate-600 dark:text-slate-400 ml-2 mt-1">
                                      {level.description.substring(0, 150)}
                                      {level.description.length > 150 ? '...' : ''}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Calibration Examples */}
                          {promptData.calibrationExamples && promptData.calibrationExamples.length > 0 && (
                            <div className="bg-cyan-50 dark:bg-cyan-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                                Calibration Examples ({promptData.calibrationExamples.length}):
                              </h5>
                              <div className="space-y-3">
                                {promptData.calibrationExamples.map((example, i) => (
                                  <div key={i} className="text-xs border-l-2 border-cyan-300 dark:border-cyan-700 pl-2">
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className="font-semibold text-cyan-700 dark:text-cyan-300">
                                        Example {i + 1}
                                      </span>
                                      {example.score !== null && (
                                        <span className={`px-1.5 py-0.5 rounded text-xs ${
                                          example.score >= 7 ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' :
                                          example.score >= 5 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300' :
                                          'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                                        }`}>
                                          {example.score}/10
                                        </span>
                                      )}
                                      {example.pass_fail && (
                                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                                          example.pass_fail === 'PASS'
                                            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                                            : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                                        }`}>
                                          {example.pass_fail}
                                        </span>
                                      )}
                                    </div>
                                    {example.analysis && (
                                      <div className="text-slate-600 dark:text-slate-400 mb-1">
                                        {example.analysis.substring(0, 120)}
                                        {example.analysis.length > 120 ? '...' : ''}
                                      </div>
                                    )}
                                    {(example.evidence_positive?.length > 0 || example.evidence_negative?.length > 0) && (
                                      <div className="mt-1 space-y-0.5">
                                        {example.evidence_positive?.length > 0 && (
                                          <div className="text-green-700 dark:text-green-300">
                                            + {example.evidence_positive[0]}
                                          </div>
                                        )}
                                        {example.evidence_negative?.length > 0 && (
                                          <div className="text-red-700 dark:text-red-300">
                                            - {example.evidence_negative[0]}
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Meta-Evaluation Details */}
                          {promptData.auditScores && Object.keys(promptData.auditScores).length > 0 && (
                            <div className="bg-slate-50 dark:bg-slate-800/50 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">Meta-Evaluation Scores:</h5>
                              <div className="grid grid-cols-5 gap-2">
                                {Object.entries(promptData.auditScores).map(([key, score]) => (
                                  <div key={key} className="text-center">
                                    <div className={`text-sm font-bold ${
                                      score >= 8 ? 'text-green-600 dark:text-green-400' :
                                      score >= 6 ? 'text-yellow-600 dark:text-yellow-400' :
                                      'text-red-600 dark:text-red-400'
                                    }`}>
                                      {score.toFixed(1)}
                                    </div>
                                    <div className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                                      {key.replace(/_/g, ' ')}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Rationale */}
                          {promptData.rationale && (
                            <div className="bg-blue-50 dark:bg-blue-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-blue-600 dark:text-blue-400 mb-1">Executive Summary:</h5>
                              <p className="text-sm text-slate-700 dark:text-slate-300 whitespace-pre-line">
                                {promptData.rationale}
                              </p>
                            </div>
                          )}

                          {/* Logic Gaps (if any) */}
                          {promptData.logicGaps && promptData.logicGaps.length > 0 && (
                            <div className="bg-amber-50 dark:bg-amber-900/20 px-4 py-3 border-b border-slate-300 dark:border-slate-700">
                              <h5 className="text-xs font-semibold text-amber-600 dark:text-amber-400 mb-2">Logic Gaps Detected:</h5>
                              <ul className="space-y-1">
                                {promptData.logicGaps.map((gap, i) => (
                                  <li key={i} className="text-xs text-amber-700 dark:text-amber-300 flex items-start gap-2">
                                    <span className="text-amber-600 dark:text-amber-400">⚠</span>
                                    <span><strong>{gap.category}:</strong> {gap.description}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Eval Prompt */}
                          <div className="p-4">
                            <div className="flex items-center justify-between mb-2">
                              <Label className="text-xs font-semibold">Evaluation Prompt:</Label>
                              <div className="flex gap-2">
                                <Button
                                  onClick={() => {
                                    navigator.clipboard.writeText(promptData.evalPrompt);
                                    toast({
                                      title: "Copied!",
                                      description: "Eval prompt copied to clipboard"
                                    });
                                  }}
                                  variant="outline"
                                  size="sm"
                                >
                                  <Download className="w-3 h-3 mr-1" />
                                  Copy
                                </Button>
                                <Button
                                  onClick={async () => {
                                    try {
                                      await fetch(`${API}/projects/${projectId}/eval-prompt/versions`, {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({
                                          eval_prompt_text: promptData.evalPrompt,
                                          changes_made: `Saved aspect: ${promptData.aspect}`,
                                          rationale: promptData.rationale || `Evaluation prompt for ${promptData.aspect}`
                                        })
                                      });
                                      await fetchEvalPromptVersions();
                                      toast({
                                        title: "Saved as Version!",
                                        description: `${promptData.aspect} saved to version history`
                                      });
                                    } catch (error) {
                                      toast({
                                        title: "Save Failed",
                                        description: "Could not save to version history",
                                        variant: "destructive"
                                      });
                                    }
                                  }}
                                  variant="default"
                                  size="sm"
                                  className="bg-purple-600 hover:bg-purple-700 text-white"
                                >
                                  <Save className="w-3 h-3 mr-1" />
                                  Save as Version
                                </Button>
                                <Button
                                  onClick={async () => {
                                    try {
                                      // Save as version first
                                      await fetch(`${API}/projects/${projectId}/eval-prompt/versions`, {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({
                                          eval_prompt_text: promptData.evalPrompt,
                                          changes_made: `Saved aspect: ${promptData.aspect}`,
                                          rationale: promptData.rationale || `Evaluation prompt for ${promptData.aspect}`
                                        })
                                      });
                                      await fetchEvalPromptVersions();

                                      // Expand dataset section
                                      setExpandedSections(prev => ({
                                        ...prev,
                                        dataset: true
                                      }));

                                      // Scroll to dataset section
                                      setTimeout(() => {
                                        const datasetSection = document.querySelector('[data-section="dataset"]');
                                        if (datasetSection) {
                                          datasetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                        }
                                      }, 100);

                                      toast({
                                        title: "Ready for Testing!",
                                        description: `${promptData.aspect} saved. Generate test dataset below.`
                                      });
                                    } catch (error) {
                                      toast({
                                        title: "Save Failed",
                                        description: "Could not save to version history",
                                        variant: "destructive"
                                      });
                                    }
                                  }}
                                  variant="default"
                                  size="sm"
                                  className="bg-blue-600 hover:bg-blue-700 text-white"
                                >
                                  <ArrowUpDown className="w-3 h-3 mr-1" />
                                  Continue to Dataset
                                </Button>
                              </div>
                            </div>
                            <Textarea
                              value={promptData.evalPrompt}
                              readOnly
                              rows={12}
                              className="font-mono text-xs bg-white dark:bg-slate-900"
                            />
                          </div>
                          </CollapsibleContent>
                        </Collapsible>
                      ))}

                      <Button
                        onClick={openModalAndFetchSuggestions}
                        variant="outline"
                        className="w-full"
                      >
                        <Plus className="w-4 h-4 mr-2" />
                        Generate More Eval Prompts
                      </Button>
                    </div>
                  )}

                  {!evalPrompt && multipleEvalPrompts.length === 0 ? (
                    <Button
                      onClick={openModalAndFetchSuggestions}
                      disabled={isAgenticGeneratingEval}
                      className="w-full bg-purple-600 hover:bg-purple-700 text-white"
                      title="Generate evaluation prompt with failure mode analysis"
                    >
                      {isAgenticGeneratingEval ? (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4 mr-2" />
                          Generate Eval Prompt
                        </>
                      )}
                    </Button>
                  ) : evalPrompt ? (
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

                      {/* Review Recommendations Notice */}
                      {evalPrompt && (
                        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-700 rounded-lg p-4">
                          <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 mt-0.5">
                              <svg className="w-5 h-5 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            </div>
                            <div className="flex-1">
                              <h3 className="font-semibold text-amber-600 dark:text-amber-400 mb-2">Review Before Using</h3>
                              <p className="text-sm text-amber-700 dark:text-amber-300 mb-2">
                                <strong>Priority areas to customize for your use case:</strong>
                              </p>
                              <ul className="text-sm text-amber-700 dark:text-amber-300 space-y-1 ml-4">
                                <li className="flex items-start gap-2">
                                  <span className="text-amber-600 dark:text-amber-400 font-bold">•</span>
                                  <span><strong>Calibration examples</strong> (most important) - Add 3-5 examples with diverse scores for consistency</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <span className="text-amber-600 dark:text-amber-400 font-bold">•</span>
                                  <span><strong>Dimension weights</strong> - Adjust if defaults don't match your priorities</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <span className="text-amber-600 dark:text-amber-400 font-bold">•</span>
                                  <span><strong>Auto-fail conditions</strong> - Add domain-specific critical failures</span>
                                </li>
                              </ul>
                              <p className="text-xs text-amber-600 dark:text-amber-400 mt-2 italic">
                                The rest can be used as-is (structure, rubrics, and consistency rules are AI-optimized)
                              </p>
                            </div>
                          </div>
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
                          onClick={handleRegenerateEvalPrompt}
                          disabled={isRegeneratingEval || isAgenticGeneratingEval}
                          className="border-green-600 text-green-600 dark:text-green-400 hover:bg-green-900/20"
                        >
                          {isRegeneratingEval ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Regenerating...
                            </>
                          ) : (
                            <>
                              <RotateCcw className="w-4 h-4 mr-2" />
                              Regenerate
                            </>
                          )}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => setShowEvalFeedback(true)}
                          className="border-blue-600 text-blue-600 dark:text-blue-400 hover:bg-blue-900/20"
                        >
                          <MessageSquare className="w-4 h-4 mr-2" />
                          Review & Refine
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => setShowEvalTest(!showEvalTest)}
                          className="border-orange-600 text-orange-600 dark:text-orange-400 hover:bg-orange-900/20"
                        >
                          <PlayCircle className="w-4 h-4 mr-2" />
                          Test Eval Prompt
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

                      {/* Eval Prompt Testing UI */}
                      {showEvalTest && (
                        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-300 dark:border-orange-700 rounded-lg p-4 space-y-4 mt-4">
                          <div className="flex items-center justify-between">
                            <h3 className="font-semibold text-orange-600 dark:text-orange-400 flex items-center gap-2">
                              <PlayCircle className="w-4 h-4" />
                              Test Your Evaluation Prompt
                            </h3>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setShowEvalTest(false);
                                setEvalTestResult(null);
                              }}
                              className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                            >
                              <X className="w-4 h-4" />
                            </Button>
                          </div>
                          <p className="text-sm text-slate-600 dark:text-slate-400">
                            Test your eval prompt with a sample input/output pair to ensure it produces valid scores.
                          </p>

                          <div className="grid gap-4">
                            <div>
                              <Label className="text-slate-700 dark:text-slate-300">Sample User Input</Label>
                              <Textarea
                                value={evalTestInput}
                                onChange={(e) => setEvalTestInput(e.target.value)}
                                placeholder="Enter a sample user input that would be sent to your system prompt..."
                                className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[80px]"
                              />
                            </div>
                            <div>
                              <Label className="text-slate-700 dark:text-slate-300">Sample AI Output</Label>
                              <Textarea
                                value={evalTestOutput}
                                onChange={(e) => setEvalTestOutput(e.target.value)}
                                placeholder="Enter a sample AI response to evaluate..."
                                className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 min-h-[80px]"
                              />
                            </div>
                            <div>
                              <Label className="text-slate-700 dark:text-slate-300">Expected Score (optional, 1-5)</Label>
                              <Input
                                type="number"
                                min="1"
                                max="5"
                                value={evalTestExpectedScore}
                                onChange={(e) => setEvalTestExpectedScore(e.target.value)}
                                placeholder="Enter expected score to validate..."
                                className="bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 w-32"
                              />
                            </div>
                          </div>

                          <Button
                            onClick={handleTestEvalPrompt}
                            disabled={isTestingEval || !evalTestInput.trim() || !evalTestOutput.trim()}
                            className="bg-orange-600 hover:bg-orange-700 text-white"
                          >
                            {isTestingEval ? (
                              <>
                                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                Testing...
                              </>
                            ) : (
                              <>
                                <PlayCircle className="w-4 h-4 mr-2" />
                                Run Test
                              </>
                            )}
                          </Button>

                          {/* Test Results */}
                          {evalTestResult && (
                            <div className={`rounded-lg p-4 space-y-3 ${
                              evalTestResult.success
                                ? 'bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700'
                                : 'bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700'
                            }`}>
                              <div className="flex items-center justify-between">
                                <h4 className={`font-semibold ${
                                  evalTestResult.success
                                    ? 'text-green-600 dark:text-green-400'
                                    : 'text-red-600 dark:text-red-400'
                                }`}>
                                  {evalTestResult.success ? 'Test Passed' : 'Test Failed'}
                                </h4>
                                {evalTestResult.score && (
                                  <span className="text-2xl font-bold text-slate-700 dark:text-slate-200">
                                    {evalTestResult.score}/5
                                  </span>
                                )}
                              </div>

                              {evalTestResult.reasoning && (
                                <div>
                                  <Label className="text-slate-700 dark:text-slate-300">Reasoning</Label>
                                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                                    {evalTestResult.reasoning}
                                  </p>
                                </div>
                              )}

                              {evalTestResult.validation && (
                                <div className="flex flex-wrap gap-2">
                                  <span className={`px-2 py-1 rounded text-xs ${
                                    evalTestResult.validation.score_found
                                      ? 'bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300'
                                      : 'bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-300'
                                  }`}>
                                    Score: {evalTestResult.validation.score_found ? 'Found' : 'Missing'}
                                  </span>
                                  <span className={`px-2 py-1 rounded text-xs ${
                                    evalTestResult.validation.reasoning_found
                                      ? 'bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300'
                                      : 'bg-yellow-100 dark:bg-yellow-800 text-yellow-700 dark:text-yellow-300'
                                  }`}>
                                    Reasoning: {evalTestResult.validation.reasoning_found ? 'Found' : 'Missing'}
                                  </span>
                                  {evalTestResult.validation.expected_score && (
                                    <span className={`px-2 py-1 rounded text-xs ${
                                      evalTestResult.validation.score_match
                                        ? 'bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300'
                                        : evalTestResult.validation.score_close
                                          ? 'bg-yellow-100 dark:bg-yellow-800 text-yellow-700 dark:text-yellow-300'
                                          : 'bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-300'
                                    }`}>
                                      Expected: {evalTestResult.validation.expected_score}, Got: {evalTestResult.validation.actual_score}
                                    </span>
                                  )}
                                </div>
                              )}

                              {evalTestResult.suggestions && evalTestResult.suggestions.length > 0 && (
                                <div>
                                  <Label className="text-slate-700 dark:text-slate-300">Suggestions</Label>
                                  <ul className="text-sm text-slate-600 dark:text-slate-400 mt-1 space-y-1">
                                    {evalTestResult.suggestions.map((suggestion, idx) => (
                                      <li key={idx} className="flex items-start gap-2">
                                        <span className="text-orange-500">•</span>
                                        <span>{suggestion}</span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}

                              {evalTestResult.error && (
                                <div className="text-sm text-red-600 dark:text-red-400">
                                  Error: {evalTestResult.error}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Eval Prompt Version History */}
                      {evalPromptVersions.length >= 1 && (
                        <div className="mt-6">
                          <div className="flex justify-between items-center mb-2">
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-slate-100">Eval Prompt Version History</h3>
                              <p className="text-xs text-slate-500 dark:text-slate-400">Click on a version to load it, or select versions to compare</p>
                            </div>
                            {selectedEvalVersionsForCompare.length >= 2 && (
                              <Button
                                size="sm"
                                onClick={() => {
                                  setEvalVersionCompareModalOpen(true);
                                  if (selectedEvalVersionsForCompare.length === 2) {
                                    const sorted = [...selectedEvalVersionsForCompare].sort((a, b) => a - b);
                                    fetchEvalVersionDiff(sorted[0], sorted[1]);
                                  }
                                }}
                                className="bg-purple-600 hover:bg-purple-700 text-white"
                              >
                                <ArrowUpDown className="w-4 h-4 mr-1" />
                                Compare ({selectedEvalVersionsForCompare.length})
                              </Button>
                            )}
                          </div>
                          <div className="space-y-2">
                            {evalPromptVersions.map((v, idx) => (
                              <div
                                key={idx}
                                onClick={() => {
                                  if (v.eval_prompt_text !== evalPrompt) {
                                    handleRestoreEvalVersion(v.version);
                                  }
                                }}
                                className={`p-3 rounded-lg border transition-all cursor-pointer ${
                                  v.eval_prompt_text === evalPrompt
                                    ? 'bg-blue-100 dark:bg-blue-900/20 border-blue-600 dark:border-blue-700'
                                    : 'bg-slate-50 dark:bg-slate-800 border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-slate-100 dark:hover:bg-slate-700'
                                }`}
                              >
                                <div className="flex justify-between items-center">
                                  <div className="flex items-center gap-3">
                                    <input
                                      type="checkbox"
                                      checked={selectedEvalVersionsForCompare.includes(v.version)}
                                      onChange={(e) => {
                                        e.stopPropagation();
                                        if (e.target.checked) {
                                          setSelectedEvalVersionsForCompare(prev => [...prev, v.version]);
                                        } else {
                                          setSelectedEvalVersionsForCompare(prev => prev.filter(ver => ver !== v.version));
                                        }
                                      }}
                                      onClick={(e) => e.stopPropagation()}
                                      className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 text-purple-600 focus:ring-purple-500"
                                    />
                                    <div className="flex items-center gap-2 flex-1">
                                      <span className="font-medium text-slate-900 dark:text-white">Version {v.version}</span>
                                      {v.eval_prompt_text === evalPrompt && (
                                        <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded">Current</span>
                                      )}
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    {v.best_practices_score !== undefined && (
                                      <span className={`text-sm font-semibold px-2 py-0.5 rounded ${
                                        v.best_practices_score >= 80
                                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                          : v.best_practices_score >= 60
                                            ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                            : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                      }`}>
                                        {v.best_practices_score}/100
                                      </span>
                                    )}
                                    {v.eval_prompt_text !== evalPrompt && evalPromptVersions.length > 1 && (
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleDeleteEvalVersion(v.version);
                                        }}
                                        className="h-6 w-6 p-0 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/20"
                                      >
                                        <Trash2 className="h-4 w-4" />
                                      </Button>
                                    )}
                                  </div>
                                </div>
                                {v.changes_made && (
                                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 ml-7">
                                    {v.changes_made}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : null}
                </CardContent>
              )}
            </Card>
          )}

          {/* Section 4: Test Dataset */}
          {projectId && (
            <Card className="bg-white dark:bg-slate-900/50 border-slate-300 dark:border-slate-700" data-section="dataset">
              <CardHeader className="p-0">
                <SectionHeader
                  section="dataset"
                  title="4. Test Dataset"
                  description="Generate test cases to evaluate your system prompt"
                />
              </CardHeader>
              {expandedSections.dataset && (
                <CardContent className="p-6 space-y-4">
                  {/* Hidden file input for CSV upload - always rendered */}
                  <input
                    type="file"
                    ref={csvUploadRef}
                    onChange={handleUploadDataset}
                    accept=".csv"
                    className="hidden"
                  />

                  {/* Sample count input */}
                  {!dataset && (
                    <div>
                      <Label htmlFor="sampleCount" className="text-slate-700 dark:text-slate-300">Number of Samples (max 100)</Label>
                      <Input
                        id="sampleCount"
                        type="number"
                        value={sampleCount}
                        onChange={(e) => setSampleCount(Math.min(parseInt(e.target.value) || 10, 100))}
                        min="10"
                        max="100"
                        className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100"
                        disabled={isGeneratingDataset}
                      />
                      <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                        Default distribution: 25% positive, 20% edge cases, 20% negative, 17% adversarial, 18% prompt injection
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
                  {!dataset && !isGeneratingDataset && !isUploadingDataset && (
                    <div className="space-y-3">
                      <div className="flex gap-2">
                        <Button
                          onClick={() => handleGenerateDataset()}
                          className="flex-1 bg-purple-600 hover:bg-purple-700 text-white"
                        >
                          <Sparkles className="w-4 h-4 mr-2" />
                          Generate Dataset
                        </Button>
                        <Button
                          onClick={() => csvUploadRef.current?.click()}
                          variant="outline"
                          className="flex-1 border-slate-400 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                        >
                          <Upload className="w-4 h-4 mr-2" />
                          Upload CSV
                        </Button>
                      </div>
                      <p className="text-xs text-slate-500 dark:text-slate-400 text-center">
                        Generate AI test cases automatically, or upload your own CSV dataset
                      </p>
                    </div>
                  )}

                  {/* Uploading indicator */}
                  {isUploadingDataset && (
                    <div className="flex items-center justify-center gap-2 py-4">
                      <RefreshCw className="w-5 h-5 animate-spin text-blue-500" />
                      <span className="text-slate-600 dark:text-slate-400">Uploading and processing CSV...</span>
                    </div>
                  )}

                  {/* Dataset generated success message and preview */}
                  {dataset && (
                    <>
                      <div className={`${
                        dataset.metadata?.generation_type === 'uploaded' ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700' :
                        dataset.metadata?.generation_type === 'smart' ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700' :
                        'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700'
                      } border rounded-lg p-4`}>
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className={`font-semibold ${
                              dataset.metadata?.generation_type === 'uploaded' ? 'text-blue-600 dark:text-blue-400' :
                              dataset.metadata?.generation_type === 'smart' ? 'text-purple-600 dark:text-purple-400' :
                              'text-green-600 dark:text-green-400'
                            } mb-2 flex items-center gap-2`}>
                              {dataset.metadata?.generation_type === 'uploaded' && <Upload className="w-4 h-4" />}
                              {dataset.metadata?.generation_type === 'smart' && <Sparkles className="w-4 h-4" />}
                              {dataset.metadata?.generation_type === 'uploaded' ? 'Dataset Uploaded' :
                               dataset.metadata?.generation_type === 'smart' ? 'Smart Dataset Generated' : 'Dataset Generated'}
                            </h3>
                            <p className="text-sm text-slate-700 dark:text-slate-300">
                              {dataset.sample_count} test cases {dataset.metadata?.generation_type === 'uploaded' ? 'imported' : 'created'}
                              {dataset.metadata?.source_file && (
                                <span className="text-slate-500 dark:text-slate-400">
                                  {' '}• From: {dataset.metadata.source_file}
                                </span>
                              )}
                              {dataset.metadata?.input_type && dataset.metadata.input_type !== 'custom' && (
                                <span className="text-slate-500 dark:text-slate-400">
                                  {' '}• Input type: {dataset.metadata.input_type.replace(/_/g, ' ')}
                                </span>
                              )}
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
                        <div className="overflow-auto max-h-96 border border-slate-300 dark:border-slate-700 rounded">
                          <table className="text-sm min-w-full">
                            <thead className="bg-slate-800 dark:bg-slate-800 sticky top-0">
                              <tr>
                                <th className="p-2 text-left text-white min-w-[300px]">Input</th>
                                <th className="p-2 text-left text-white min-w-[100px]">Category</th>
                                <th className="p-2 text-left text-white min-w-[200px]">Test Focus</th>
                                <th className="p-2 text-left text-white min-w-[100px]">Difficulty</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dataset.preview?.map((test, idx) => (
                                <tr key={idx} className="border-t border-slate-300 dark:border-slate-700">
                                  <td className="p-2 text-slate-900 dark:text-slate-100 whitespace-nowrap" title={getTestInputDisplay(test, 500)}>{getTestInputDisplay(test, 60)}...</td>
                                  <td className="p-2 whitespace-nowrap">
                                    <span className={`px-2 py-1 rounded text-xs text-white ${
                                      test.category === 'positive' ? 'bg-green-600' :
                                      test.category === 'edge_case' ? 'bg-yellow-600' :
                                      test.category === 'negative' ? 'bg-red-600' :
                                      'bg-purple-600'
                                    }`}>
                                      {test.category}
                                    </span>
                                  </td>
                                  <td className="p-2 text-slate-600 dark:text-slate-400 whitespace-nowrap">{test.test_focus}</td>
                                  <td className="p-2 text-slate-600 dark:text-slate-400 whitespace-nowrap">{test.difficulty}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      <div className="flex gap-2 flex-wrap">
                        <Button onClick={handleDownloadDataset} variant="outline" className="border-slate-400 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800">
                          <Download className="w-4 h-4 mr-2" />
                          Download
                        </Button>
                        <Button
                          onClick={() => csvUploadRef.current?.click()}
                          variant="outline"
                          disabled={isUploadingDataset}
                          className="border-blue-500 text-blue-600 dark:text-blue-400 hover:bg-blue-900/20"
                        >
                          {isUploadingDataset ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Uploading...
                            </>
                          ) : (
                            <>
                              <Upload className="w-4 h-4 mr-2" />
                              Upload CSV
                            </>
                          )}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={handleRegenerateDatasetNew}
                          disabled={isRegeneratingDataset || isGeneratingDataset}
                          className="border-green-600 text-green-600 dark:text-green-400 hover:bg-green-900/20"
                        >
                          {isRegeneratingDataset ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Regenerating...
                            </>
                          ) : (
                            <>
                              <RotateCcw className="w-4 h-4 mr-2" />
                              Regenerate
                            </>
                          )}
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
                title={<>5. Test Run <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400 rounded-full">BETA</span></>}
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
                  <div className="space-y-4">
                    {/* Row 1: Prompt Version and Pass Threshold */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <Label className="text-slate-700 dark:text-slate-300">Prompt Version</Label>
                        <Select
                          value={String(selectedTestRunVersion || currentVersion?.version || 1)}
                          onValueChange={(v) => setSelectedTestRunVersion(parseInt(v))}
                        >
                          <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                            <SelectValue placeholder="Select version" />
                          </SelectTrigger>
                          <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                            {versionHistory.map((v) => (
                              <SelectItem key={v.version} value={String(v.version)} className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700">
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
                          <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                            <SelectItem value="3.0" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">3.0 (Lenient)</SelectItem>
                            <SelectItem value="3.5" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">3.5 (Standard)</SelectItem>
                            <SelectItem value="4.0" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">4.0 (Strict)</SelectItem>
                            <SelectItem value="4.5" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">4.5 (Very Strict)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label className="text-slate-700 dark:text-slate-300">Response Model</Label>
                        <Select value={llmModel || modelOptions[llmProvider]?.[0] || ""} onValueChange={setLlmModel}>
                          <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                            <SelectValue placeholder="Select model" />
                          </SelectTrigger>
                          <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                            {modelOptions[llmProvider]?.map((model, index) => {
                              if (typeof model === 'object' && model.disabled) {
                                return (
                                  <SelectItem key={index} value={`disabled-${index}`} disabled className="text-slate-500 dark:text-slate-400 font-semibold">
                                    {model.label}
                                  </SelectItem>
                                );
                              }
                              return (
                                <SelectItem key={model} value={model} className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700">
                                  {model}
                                </SelectItem>
                              );
                            })}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {/* Row 2: Evaluation Model Configuration */}
                    <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Label className="text-slate-700 dark:text-slate-300 font-medium">Use Separate Evaluation Model</Label>
                          <span className="text-xs text-slate-500 dark:text-slate-400">(Recommended: Use thinking model for better evaluation)</span>
                        </div>
                        <button
                          type="button"
                          onClick={() => setUseSeparateEvalModel(!useSeparateEvalModel)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            useSeparateEvalModel ? 'bg-blue-600' : 'bg-slate-300 dark:bg-slate-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              useSeparateEvalModel ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      {useSeparateEvalModel && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <Label className="text-slate-600 dark:text-slate-400 text-sm">Eval Provider</Label>
                            <Select value={evalProvider} onValueChange={(v) => {
                              setEvalProvider(v);
                              setEvalModel(thinkingModels[v]?.[0] || "");
                            }}>
                              <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                                <SelectItem value="openai" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700">OpenAI</SelectItem>
                                <SelectItem value="claude" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700">Anthropic</SelectItem>
                                <SelectItem value="gemini" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700">Google</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div>
                            <Label className="text-slate-600 dark:text-slate-400 text-sm">Eval Model (Thinking/Reasoning)</Label>
                            <Select value={evalModel} onValueChange={setEvalModel}>
                              <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                                <SelectValue placeholder="Select thinking model" />
                              </SelectTrigger>
                              <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                                {thinkingModels[evalProvider]?.map((model) => (
                                  <SelectItem key={model} value={model} className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700">
                                    {model}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Row 3: Start Button */}
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
                            {testRunResults.summary.total_tokens?.toLocaleString() || 0}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">Total Tokens</div>
                        </div>
                      </div>

                      {/* Latency Metrics */}
                      <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-3">
                        <div className="text-center p-2 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-lg font-semibold text-amber-600">
                            {testRunResults.summary.avg_ttfb_ms || 0}ms
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">Avg TTFB</div>
                        </div>
                        <div className="text-center p-2 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-lg font-semibold text-cyan-600">
                            {testRunResults.summary.min_latency_ms || 0}ms
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">Min Latency</div>
                        </div>
                        <div className="text-center p-2 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-lg font-semibold text-indigo-600">
                            {testRunResults.summary.avg_latency_ms || 0}ms
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">Avg Latency</div>
                        </div>
                        <div className="text-center p-2 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-lg font-semibold text-rose-600">
                            {testRunResults.summary.max_latency_ms || 0}ms
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">Max Latency</div>
                        </div>
                        <div className="text-center p-2 bg-white dark:bg-slate-800 rounded-lg">
                          <div className="text-lg font-semibold text-emerald-600">
                            ${testRunResults.summary.estimated_cost?.toFixed(4) || '0.0000'}
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">Est. Cost</div>
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
                        className="text-slate-700 dark:text-slate-300"
                      >
                        <Download className="w-4 h-4 mr-1" />
                        Export CSV
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleExportResults("json")}
                        className="text-slate-700 dark:text-slate-300"
                      >
                        <FileText className="w-4 h-4 mr-1" />
                        Export JSON
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSingleTestOpen(true)}
                        className="text-slate-700 dark:text-slate-300"
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
                          className="text-slate-700 dark:text-slate-300"
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
                        className={`text-slate-700 dark:text-slate-300 ${compareMode ? 'bg-blue-100 dark:bg-blue-900/30' : ''}`}
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
                            <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                              <SelectValue placeholder="Select run..." />
                            </SelectTrigger>
                            <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                              {testRunHistory.map((run) => (
                                <SelectItem key={run.run_id} value={run.run_id} className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">
                                  V{run.prompt_version} - {run.summary?.pass_rate || 0}% pass
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label className="text-slate-700 dark:text-slate-300">Run B (Compare)</Label>
                          <Select value={compareRunB || ""} onValueChange={setCompareRunB}>
                            <SelectTrigger className="mt-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                              <SelectValue placeholder="Select run..." />
                            </SelectTrigger>
                            <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                              {testRunHistory.map((run) => (
                                <SelectItem key={run.run_id} value={run.run_id} className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">
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
                                  {(result.dataset_item_index ?? result.index ?? 0) + 1}
                                </td>
                                <td className="p-2 text-slate-900 dark:text-slate-100 max-w-xs truncate">
                                  {getTestInputDisplay(result.input_data || result, 80)}...
                                </td>
                                <td className="p-2 text-slate-900 dark:text-slate-100 max-w-xs truncate">
                                  {String(result.prompt_output || result.output || '').substring(0, 100)}...
                                </td>
                                <td className="p-2">
                                  <span className={`px-2 py-1 rounded text-xs font-bold ${
                                    (result.eval_score ?? result.score ?? 0) >= 4 ? 'bg-green-600 text-white' :
                                    (result.eval_score ?? result.score ?? 0) >= 3 ? 'bg-yellow-600 text-white' :
                                    'bg-red-600 text-white'
                                  }`}>
                                    {(result.eval_score ?? result.score ?? 0).toFixed(1)}
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
                                  {String(result.eval_feedback || result.feedback || '').substring(0, 150)}...
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
                        Previous Test Runs ({testRunHistory.length})
                      </h3>
                      <div className="space-y-2">
                        {testRunHistory.map((run, index) => (
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
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-bold text-white bg-slate-500 dark:bg-slate-600 px-2 py-0.5 rounded">
                                  #{testRunHistory.length - index}
                                </span>
                                <span className="font-medium text-slate-900 dark:text-slate-100">
                                  Prompt v{run.prompt_version || run.version_number || '?'}
                                </span>
                                {run.model_name && (
                                  <span className="text-xs text-slate-500 dark:text-slate-400">
                                    ({run.model_name})
                                  </span>
                                )}
                                <span className={`px-2 py-0.5 rounded text-xs ${
                                  run.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                                  run.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                                  'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300'
                                }`}>
                                  {run.status}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-sm text-slate-600 dark:text-slate-400">
                                {run.summary && (
                                  <span className="font-medium">
                                    Pass: {run.summary.pass_rate}%
                                  </span>
                                )}
                                <span className="text-xs">
                                  {new Date(run.created_at).toLocaleDateString()} {new Date(run.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                </span>
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
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-900 dark:text-slate-100">
              Test Result #{(detailViewItem?.dataset_item_index ?? detailViewItem?.test_case_id ?? 0) + 1}
              {detailViewItem?.passed ? (
                <span className="px-2 py-0.5 bg-green-600 text-white text-xs rounded">PASS</span>
              ) : (
                <span className="px-2 py-0.5 bg-red-600 text-white text-xs rounded">FAIL</span>
              )}
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                (detailViewItem?.eval_score ?? detailViewItem?.score ?? 0) >= 4 ? 'bg-green-600 text-white' :
                (detailViewItem?.eval_score ?? detailViewItem?.score ?? 0) >= 3 ? 'bg-yellow-600 text-white' :
                'bg-red-600 text-white'
              }`}>
                Score: {(detailViewItem?.eval_score ?? detailViewItem?.score ?? 0).toFixed(1)}
              </span>
            </DialogTitle>
          </DialogHeader>

          {detailViewItem && (
            <div className="space-y-4">
              <div>
                <Label className="text-amber-600 dark:text-amber-400 font-semibold">Input</Label>
                <div className="mt-1 p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg whitespace-pre-wrap text-sm max-h-32 overflow-y-auto text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-600">
                  {getTestInputDisplay(detailViewItem.input_data || detailViewItem, 5000) || 'N/A'}
                </div>
              </div>

              <div>
                <Label className="text-emerald-600 dark:text-emerald-400 font-semibold">Prompt Output</Label>
                <div className="mt-1 p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg whitespace-pre-wrap text-sm max-h-48 overflow-y-auto text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-600">
                  {detailViewItem.prompt_output || detailViewItem.output || 'N/A'}
                </div>
              </div>

              <div>
                <Label className="text-blue-600 dark:text-blue-400 font-semibold">Evaluation Feedback</Label>
                <div className="mt-1 p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg whitespace-pre-wrap text-sm max-h-48 overflow-y-auto text-slate-800 dark:text-slate-200 border border-blue-200 dark:border-blue-800">
                  {detailViewItem.eval_feedback || detailViewItem.feedback || 'N/A'}
                </div>
              </div>

              <div className="grid grid-cols-4 gap-4 pt-4 border-t border-slate-200 dark:border-slate-600">
                <div>
                  <Label className="text-slate-500 dark:text-slate-400 text-xs">TTFB</Label>
                  <div className="font-mono text-slate-800 dark:text-slate-200">{detailViewItem.ttfb_ms || 0}ms</div>
                </div>
                <div>
                  <Label className="text-slate-500 dark:text-slate-400 text-xs">Total Latency</Label>
                  <div className="font-mono text-slate-800 dark:text-slate-200">{detailViewItem.latency_ms || 0}ms</div>
                </div>
                <div>
                  <Label className="text-slate-500 dark:text-slate-400 text-xs">Tokens Used</Label>
                  <div className="font-mono text-slate-800 dark:text-slate-200">{detailViewItem.tokens_used?.toLocaleString() || 0}</div>
                </div>
                <div>
                  <Label className="text-slate-500 dark:text-slate-400 text-xs">Error</Label>
                  <div className={detailViewItem.error ? 'text-red-500' : 'text-green-500'}>
                    {detailViewItem.error ? String(detailViewItem.error) : 'None'}
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
                <SelectTrigger className="mt-2 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                  <SelectItem value="10" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">10 (Quick test)</SelectItem>
                  <SelectItem value="25" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">25 (Small)</SelectItem>
                  <SelectItem value="50" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">50 (Medium)</SelectItem>
                  <SelectItem value="100" className="text-slate-900 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 focus:bg-slate-100 dark:focus:bg-slate-700 data-[highlighted]:bg-slate-100 dark:data-[highlighted]:bg-slate-700 data-[highlighted]:text-slate-900 dark:data-[highlighted]:text-slate-100">100 (Max)</SelectItem>
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
            <Button onClick={handleRegenerateDataset} className="bg-purple-600 hover:bg-purple-700 text-white">
              <Sparkles className="w-4 h-4 mr-2" />
              Generate {regenerateSampleCount} Cases
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Version Comparison Modal */}
      <Dialog open={versionCompareModalOpen} onOpenChange={(open) => {
        setVersionCompareModalOpen(open);
        if (!open) {
          setSelectedVersionsForCompare([]);
          setVersionDiffData(null);
        }
      }}>
        <DialogContent className="max-w-[95vw] w-full max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ArrowUpDown className="w-5 h-5" />
              Compare Prompt Versions
            </DialogTitle>
            <DialogDescription className="flex items-center justify-between">
              <span>Side-by-side comparison of {selectedVersionsForCompare.length} selected versions</span>
              {selectedVersionsForCompare.length === 2 && (
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant={diffViewMode === "side-by-side" ? "default" : "outline"}
                    onClick={() => setDiffViewMode("side-by-side")}
                    className="text-xs h-7"
                  >
                    Side by Side
                  </Button>
                  <Button
                    size="sm"
                    variant={diffViewMode === "unified" ? "default" : "outline"}
                    onClick={() => setDiffViewMode("unified")}
                    className="text-xs h-7"
                  >
                    Unified Diff
                  </Button>
                </div>
              )}
            </DialogDescription>
          </DialogHeader>

          {/* Diff Stats Banner */}
          {versionDiffData && selectedVersionsForCompare.length === 2 && (
            <div className="flex items-center gap-4 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-lg text-sm">
              <span className="text-slate-600 dark:text-slate-400">
                Similarity: <span className="font-semibold text-slate-900 dark:text-white">{versionDiffData.similarity_percent}%</span>
              </span>
              <span className="text-green-600 dark:text-green-400">
                +{versionDiffData.stats?.added || 0} added
              </span>
              <span className="text-red-600 dark:text-red-400">
                -{versionDiffData.stats?.removed || 0} removed
              </span>
              <span className="text-slate-500 dark:text-slate-400">
                {versionDiffData.stats?.unchanged || 0} unchanged
              </span>
            </div>
          )}

          <div className="flex-1 overflow-auto py-4">
            {/* Unified Diff View (for 2 versions) */}
            {selectedVersionsForCompare.length === 2 && diffViewMode === "unified" && versionDiffData ? (
              <div className="border rounded-lg overflow-hidden bg-white dark:bg-slate-800">
                <div className="p-3 bg-slate-100 dark:bg-slate-700 border-b flex justify-between items-center">
                  <span className="font-semibold text-slate-900 dark:text-white">
                    Version {versionDiffData.version_a} → Version {versionDiffData.version_b}
                  </span>
                </div>
                <div className="p-3 overflow-auto font-mono text-sm">
                  {isLoadingDiff ? (
                    <div className="flex items-center justify-center py-8">
                      <RefreshCw className="w-5 h-5 animate-spin text-slate-400" />
                      <span className="ml-2 text-slate-500">Loading diff...</span>
                    </div>
                  ) : (
                    versionDiffData.diff_lines?.map((line, idx) => (
                      <div
                        key={idx}
                        className={`py-0.5 px-2 -mx-2 ${
                          line.type === 'added'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                            : line.type === 'removed'
                              ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                              : 'text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        <span className="inline-block w-8 text-slate-400 select-none">
                          {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
                        </span>
                        <span className="whitespace-pre-wrap">
                          {line.content_a || line.content_b || ''}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : (
              /* Side-by-Side View */
              <div className={`grid gap-4 ${selectedVersionsForCompare.length === 2 ? 'grid-cols-2' : selectedVersionsForCompare.length === 3 ? 'grid-cols-3' : 'grid-cols-2 lg:grid-cols-4'}`}>
                {selectedVersionsForCompare
                  .sort((a, b) => a - b)
                  .map((versionNum, colIdx) => {
                    const version = versionHistory.find(v => v.version === versionNum);
                    if (!version) return null;
                    const avgScore = version.evaluation
                      ? ((version.evaluation.requirements_alignment || 0) + (version.evaluation.best_practices_score || 0)) / 2
                      : null;

                    // For side-by-side diff highlighting (when 2 versions)
                    const isLeftColumn = colIdx === 0;
                    const diffLines = versionDiffData?.diff_lines || [];

                    return (
                      <div key={versionNum} className="flex flex-col h-full border rounded-lg overflow-hidden bg-white dark:bg-slate-800">
                        {/* Version Header */}
                        <div className="p-3 bg-slate-100 dark:bg-slate-700 border-b border-slate-200 dark:border-slate-600">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-slate-900 dark:text-white">Version {version.version}</span>
                              {version.version === currentVersion?.version && (
                                <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded">Current</span>
                              )}
                            </div>
                            {avgScore !== null && (
                              <span className={`text-sm font-semibold px-2 py-0.5 rounded ${
                                avgScore >= 80
                                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                  : avgScore >= 60
                                    ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                              }`}>
                                Score: {avgScore.toFixed(1)}
                              </span>
                            )}
                          </div>
                          {version.evaluation && (
                            <div className="flex gap-4 mt-2 text-xs text-slate-600 dark:text-slate-400">
                              <span>Req: {version.evaluation.requirements_alignment || 0}%</span>
                              <span>Best Practices: {version.evaluation.best_practices_score || 0}%</span>
                            </div>
                          )}
                        </div>

                        {/* Prompt Content with Diff Highlighting */}
                        <div className="flex-1 p-3 overflow-auto">
                          {selectedVersionsForCompare.length === 2 && versionDiffData && diffLines.length > 0 ? (
                            <div className="font-mono text-sm">
                              {diffLines.map((line, idx) => {
                                // Show removed lines only in left column, added only in right
                                if (isLeftColumn && line.type === 'added') return null;
                                if (!isLeftColumn && line.type === 'removed') return null;

                                const content = isLeftColumn ? line.content_a : line.content_b;
                                if (content === null && line.type !== 'unchanged') return null;

                                return (
                                  <div
                                    key={idx}
                                    className={`py-0.5 px-1 -mx-1 ${
                                      line.type === 'added'
                                        ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                                        : line.type === 'removed'
                                          ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                                          : 'text-slate-700 dark:text-slate-300'
                                    }`}
                                  >
                                    <span className="whitespace-pre-wrap">{content || ''}</span>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <pre className="text-sm whitespace-pre-wrap font-mono text-slate-800 dark:text-slate-200 leading-relaxed">
                              {version.prompt_text}
                            </pre>
                          )}
                        </div>

                        {/* Version Footer with metadata */}
                        {version.evaluation?.suggestions && version.evaluation.suggestions.length > 0 && (
                          <div className="p-3 bg-slate-50 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-600">
                            <p className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">Suggestions:</p>
                            <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                              {version.evaluation.suggestions.slice(0, 3).map((s, i) => (
                                <li key={i} className="flex items-start gap-1">
                                  <span className="text-amber-500">•</span>
                                  <span className="line-clamp-2">{typeof s === 'string' ? s : s.suggestion}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
            )}
          </div>

          <DialogFooter className="border-t pt-4">
            <Button variant="outline" onClick={() => {
              setVersionCompareModalOpen(false);
              setSelectedVersionsForCompare([]);
              setVersionDiffData(null);
            }}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Eval Prompt Version Compare Modal */}
      <Dialog open={evalVersionCompareModalOpen} onOpenChange={(open) => {
        setEvalVersionCompareModalOpen(open);
        if (!open) {
          setSelectedEvalVersionsForCompare([]);
          setEvalVersionDiffData(null);
        }
      }}>
        <DialogContent className="max-w-[95vw] w-full max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ArrowUpDown className="w-5 h-5" />
              Compare Eval Prompt Versions
            </DialogTitle>
            <DialogDescription>
              Side-by-side comparison of {selectedEvalVersionsForCompare.length} selected eval prompt versions
            </DialogDescription>
          </DialogHeader>

          {/* Diff Stats Banner */}
          {evalVersionDiffData && selectedEvalVersionsForCompare.length === 2 && (
            <div className="flex items-center gap-4 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-lg text-sm">
              <span className="text-slate-600 dark:text-slate-400">
                Similarity: <span className="font-semibold text-slate-900 dark:text-white">{evalVersionDiffData.similarity_percent}%</span>
              </span>
              <span className="text-green-600 dark:text-green-400">
                +{evalVersionDiffData.stats?.added || 0} added
              </span>
              <span className="text-red-600 dark:text-red-400">
                -{evalVersionDiffData.stats?.removed || 0} removed
              </span>
              {evalVersionDiffData.score_a !== undefined && evalVersionDiffData.score_b !== undefined && (
                <span className="text-slate-500 dark:text-slate-400">
                  Score: {evalVersionDiffData.score_a} → {evalVersionDiffData.score_b}
                </span>
              )}
            </div>
          )}

          <div className="flex-1 overflow-auto py-4">
            {isLoadingEvalDiff ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="w-5 h-5 animate-spin text-slate-400" />
                <span className="ml-2 text-slate-500">Loading diff...</span>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                {selectedEvalVersionsForCompare
                  .sort((a, b) => a - b)
                  .map((versionNum, colIdx) => {
                    const version = evalPromptVersions.find(v => v.version === versionNum);
                    if (!version) return null;

                    const isLeftColumn = colIdx === 0;
                    const diffLines = evalVersionDiffData?.diff_lines || [];

                    return (
                      <div key={versionNum} className="flex flex-col h-full border rounded-lg overflow-hidden bg-white dark:bg-slate-800">
                        {/* Version Header */}
                        <div className="p-3 bg-slate-100 dark:bg-slate-700 border-b border-slate-200 dark:border-slate-600">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-slate-900 dark:text-white">Version {version.version}</span>
                              {version.eval_prompt_text === evalPrompt && (
                                <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded">Current</span>
                              )}
                            </div>
                            {version.best_practices_score !== undefined && (
                              <span className={`text-sm font-semibold px-2 py-0.5 rounded ${
                                version.best_practices_score >= 80
                                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                  : version.best_practices_score >= 60
                                    ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                              }`}>
                                Score: {version.best_practices_score}/100
                              </span>
                            )}
                          </div>
                          {version.changes_made && (
                            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                              {version.changes_made}
                            </p>
                          )}
                        </div>

                        {/* Prompt Content with Diff Highlighting */}
                        <div className="flex-1 p-3 overflow-auto max-h-[50vh]">
                          {diffLines.length > 0 ? (
                            <div className="font-mono text-sm">
                              {diffLines.map((line, idx) => {
                                if (isLeftColumn && line.type === 'added') return null;
                                if (!isLeftColumn && line.type === 'removed') return null;

                                const content = isLeftColumn ? line.content_a : line.content_b;
                                if (content === null && line.type !== 'unchanged') return null;

                                return (
                                  <div
                                    key={idx}
                                    className={`py-0.5 px-1 -mx-1 ${
                                      line.type === 'added'
                                        ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                                        : line.type === 'removed'
                                          ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                                          : 'text-slate-700 dark:text-slate-300'
                                    }`}
                                  >
                                    <span className="whitespace-pre-wrap">{content || ''}</span>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <pre className="text-sm whitespace-pre-wrap font-mono text-slate-800 dark:text-slate-200 leading-relaxed">
                              {version.eval_prompt_text}
                            </pre>
                          )}
                        </div>
                      </div>
                    );
                  })}
              </div>
            )}
          </div>

          <DialogFooter className="border-t pt-4">
            <Button variant="outline" onClick={() => {
              setEvalVersionCompareModalOpen(false);
              setSelectedEvalVersionsForCompare([]);
              setEvalVersionDiffData(null);
            }}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Multi-Aspect Eval Generation Modal */}
      <Dialog open={isEvalAspectsModalOpen} onOpenChange={setIsEvalAspectsModalOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>What do you want to evaluate?</DialogTitle>
            <DialogDescription>
              {isFetchingSuggestions
                ? "✨ AI is analyzing your system prompt to suggest relevant aspects..."
                : "Enter different evaluation aspects, one per line. A separate eval prompt will be generated for each aspect."}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* AI Suggestion Info */}
            {suggestionReasoning && suggestedDomain && (
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Sparkles className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5" />
                  <div>
                    <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 mb-1">
                      AI-Generated Suggestions ({suggestedDomain} Domain)
                    </p>
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      {suggestionReasoning}
                    </p>
                    <p className="text-xs text-blue-600 dark:text-blue-400 mt-1 italic">
                      You can edit, add, or remove aspects below.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="evaluation-aspects">Evaluation Aspects</Label>
              <Textarea
                id="evaluation-aspects"
                placeholder={isFetchingSuggestions
                  ? "Loading AI suggestions..."
                  : "Example:&#10;Accuracy and factual correctness&#10;Tone and professionalism&#10;Clarity and conciseness"}
                value={evaluationAspects}
                onChange={(e) => setEvaluationAspects(e.target.value)}
                rows={8}
                className="font-mono text-sm"
                disabled={isFetchingSuggestions}
              />
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {isFetchingSuggestions
                  ? "⏳ Generating smart suggestions based on your system prompt..."
                  : `Tip: Each line will generate a separate evaluation prompt. Currently ${evaluationAspects.split('\n').filter(line => line.trim().length > 0).length} aspect(s).`}
              </p>
            </div>

            {/* Display multiple eval prompts if they exist */}
            {multipleEvalPrompts.length > 0 && (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                <h4 className="font-semibold text-sm">Generated Eval Prompts:</h4>
                {multipleEvalPrompts.map((promptData, index) => (
                  <div key={index} className="border border-slate-300 dark:border-slate-700 rounded-lg p-3 space-y-2">
                    <div className="flex items-start justify-between">
                      <h5 className="font-medium text-sm text-purple-600 dark:text-purple-400">
                        {index + 1}. {promptData.aspect}
                      </h5>
                      {promptData.metaQuality && (
                        <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                          Quality: {promptData.metaQuality.toFixed(1)}/10
                        </span>
                      )}
                    </div>
                    <pre className="text-xs bg-slate-50 dark:bg-slate-800 p-2 rounded overflow-x-auto whitespace-pre-wrap">
{promptData.evalPrompt}
                    </pre>
                    {promptData.rationale && (
                      <p className="text-xs text-slate-600 dark:text-slate-400">
                        <strong>Rationale:</strong> {promptData.rationale}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsEvalAspectsModalOpen(false);
                setEvaluationAspects("");
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleMultipleEvalGeneration}
              disabled={isAgenticGeneratingEval || !evaluationAspects.trim()}
              className="bg-purple-600 hover:bg-purple-700 text-white"
            >
              {isAgenticGeneratingEval ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Generate Eval Prompts
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PromptOptimizer;
