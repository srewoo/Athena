/**
 * Settings Modal Component
 * Handles LLM provider, API key configuration, and Domain Context
 */
import React, { useState, useEffect, useRef } from "react";
import { Settings, Eye, EyeOff, Upload, FileJson, Trash2, Check, AlertCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { useToast } from "../../hooks/use-toast";
import { API } from "../../App";

// Model options per provider (2025 latest releases)
export const MODEL_OPTIONS = {
  openai: [
    { value: "gpt-4o", label: "GPT-4o (Recommended)" },
    { value: "gpt-4o-mini", label: "GPT-4o Mini" },
    { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
    { value: "---reasoning---", label: "--- Reasoning Models ---", disabled: true },
    { value: "o1", label: "O1" },
    { value: "o1-mini", label: "O1 Mini" },
    { value: "o1-preview", label: "O1 Preview" },
    { value: "o3-mini", label: "O3 Mini" },
  ],
  claude: [
    { value: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet (Recommended)" },
    { value: "claude-3-opus-20240229", label: "Claude 3 Opus" },
    { value: "claude-3-haiku-20240307", label: "Claude 3 Haiku" },
  ],
  gemini: [
    { value: "gemini-2.0-flash-exp", label: "Gemini 2.0 Flash (Recommended)" },
    { value: "gemini-1.5-pro", label: "Gemini 1.5 Pro" },
    { value: "gemini-1.5-flash", label: "Gemini 1.5 Flash" },
  ]
};

// Thinking/Reasoning models for evaluation
export const THINKING_MODELS = {
  openai: ["o1", "o1-mini", "o1-preview", "o3-mini"],
  claude: ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
  gemini: ["gemini-1.5-pro"]
};

const SettingsModal = ({
  open,
  onOpenChange,
  settings,
  onSettingsChange,
  onSave
}) => {
  const { toast } = useToast();
  const [showApiKey, setShowApiKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTab, setActiveTab] = useState("llm");

  // Domain Context state
  const [domainContext, setDomainContext] = useState(null);
  const [contextJson, setContextJson] = useState("");
  const [contextError, setContextError] = useState("");
  const [hasContext, setHasContext] = useState(false);
  const [isLoadingContext, setIsLoadingContext] = useState(false);
  const [isSavingContext, setIsSavingContext] = useState(false);
  const fileInputRef = useRef(null);

  const {
    llmProvider = "openai",
    llmModel = "",
    apiKey = "",
    evalProvider = "openai",
    evalModel = "o1-mini",
    useSeparateEvalModel = true
  } = settings || {};

  // Load domain context when modal opens
  useEffect(() => {
    if (open && activeTab === "context") {
      loadDomainContext();
    }
  }, [open, activeTab]);

  const loadDomainContext = async () => {
    setIsLoadingContext(true);
    try {
      const response = await fetch(`${API}/domain-context`);
      if (response.ok) {
        const data = await response.json();
        setDomainContext(data.context);
        setContextJson(JSON.stringify(data.context, null, 2));
        setHasContext(data.has_context);
        setContextError("");
      }
    } catch (error) {
      console.error("Failed to load domain context:", error);
    } finally {
      setIsLoadingContext(false);
    }
  };

  const handleContextJsonChange = (value) => {
    setContextJson(value);
    try {
      JSON.parse(value);
      setContextError("");
    } catch (e) {
      setContextError("Invalid JSON: " + e.message);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target.result;
        const parsed = JSON.parse(content);
        setDomainContext(parsed);
        setContextJson(JSON.stringify(parsed, null, 2));
        setContextError("");
        toast({
          title: "File Loaded",
          description: `Loaded ${file.name}. Click 'Save Context' to apply.`
        });
      } catch (error) {
        setContextError("Invalid JSON file: " + error.message);
        toast({
          title: "Invalid JSON",
          description: "The file does not contain valid JSON",
          variant: "destructive"
        });
      }
    };
    reader.readAsText(file);
    event.target.value = ""; // Reset input
  };

  const saveDomainContext = async () => {
    if (contextError) {
      toast({
        title: "Invalid JSON",
        description: "Please fix the JSON errors before saving",
        variant: "destructive"
      });
      return;
    }

    setIsSavingContext(true);
    try {
      const parsed = JSON.parse(contextJson);
      const response = await fetch(`${API}/domain-context`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed)
      });

      if (response.ok) {
        const data = await response.json();
        setHasContext(data.populated_fields > 0);
        toast({
          title: "Context Saved",
          description: `Domain context saved with ${data.populated_fields} populated fields`
        });
      } else {
        throw new Error("Failed to save");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save domain context",
        variant: "destructive"
      });
    } finally {
      setIsSavingContext(false);
    }
  };

  const clearDomainContext = async () => {
    try {
      const response = await fetch(`${API}/domain-context`, {
        method: "DELETE"
      });
      if (response.ok) {
        await loadDomainContext();
        toast({
          title: "Context Cleared",
          description: "Domain context has been reset"
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to clear domain context",
        variant: "destructive"
      });
    }
  };

  const handleSave = async () => {
    if (!apiKey.trim()) {
      toast({
        title: "API Key Required",
        description: "Please enter your API key",
        variant: "destructive"
      });
      return;
    }

    setIsSaving(true);

    // Get default model if none selected
    const modelToUse = llmModel || MODEL_OPTIONS[llmProvider]?.[0]?.value || "gpt-4o";

    const settingsData = {
      llm_provider: llmProvider,
      api_key: apiKey,
      model_name: modelToUse
    };

    try {
      const response = await fetch(`${API}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settingsData)
      });

      if (response.ok) {
        // Save to localStorage for persistence
        localStorage.setItem('athena_llm_settings', JSON.stringify(settingsData));

        toast({
          title: "Settings Saved",
          description: `Using ${llmProvider.toUpperCase()} for AI operations`
        });

        onSave && onSave(settingsData);
        onOpenChange(false);
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
      setIsSaving(false);
    }
  };

  // Get models for current provider
  const currentModels = MODEL_OPTIONS[llmProvider] || [];
  const thinkingModels = THINKING_MODELS[evalProvider] || [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Settings
          </DialogTitle>
          <DialogDescription>
            Configure LLM provider and domain context for test data generation
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="llm">LLM Settings</TabsTrigger>
            <TabsTrigger value="context" className="flex items-center gap-1">
              Domain Context
              {hasContext && <Check className="h-3 w-3 text-green-500" />}
            </TabsTrigger>
          </TabsList>

          {/* LLM Settings Tab */}
          <TabsContent value="llm" className="space-y-4 py-4">
            {/* Provider Selection */}
            <div className="space-y-2">
              <Label>LLM Provider</Label>
              <Select
                value={llmProvider}
                onValueChange={(value) => {
                  onSettingsChange({
                    ...settings,
                    llmProvider: value,
                    llmModel: MODEL_OPTIONS[value]?.[0]?.value || ""
                  });
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="claude">Anthropic Claude</SelectItem>
                  <SelectItem value="gemini">Google Gemini</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label>Model</Label>
              <Select
                value={llmModel}
                onValueChange={(value) => onSettingsChange({ ...settings, llmModel: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {currentModels.map((model) => (
                    model.disabled ? (
                      <SelectItem key={model.value} value={model.value} disabled>
                        {model.label}
                      </SelectItem>
                    ) : (
                      <SelectItem key={model.value} value={model.value}>
                        {model.label}
                      </SelectItem>
                    )
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* API Key */}
            <div className="space-y-2">
              <Label>API Key</Label>
              <div className="relative">
                <Input
                  type={showApiKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => onSettingsChange({ ...settings, apiKey: e.target.value })}
                  placeholder={`Enter your ${llmProvider} API key`}
                  className="pr-10"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Your API key is stored locally and never sent to our servers
              </p>
            </div>

            {/* Separate Eval Model */}
            <div className="space-y-2 pt-4 border-t">
              <div className="flex items-center justify-between">
                <Label>Use separate model for evaluation</Label>
                <input
                  type="checkbox"
                  checked={useSeparateEvalModel}
                  onChange={(e) => onSettingsChange({
                    ...settings,
                    useSeparateEvalModel: e.target.checked
                  })}
                  className="h-4 w-4"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Recommended: Use a reasoning model for more accurate evaluations
              </p>

              {useSeparateEvalModel && (
                <div className="space-y-2 mt-2">
                  <Label>Evaluation Model</Label>
                  <Select
                    value={evalModel}
                    onValueChange={(value) => onSettingsChange({ ...settings, evalModel: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select evaluation model" />
                    </SelectTrigger>
                    <SelectContent>
                      {thinkingModels.map((model) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>

            <DialogFooter className="pt-4">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={isSaving}>
                {isSaving ? "Saving..." : "Save Settings"}
              </Button>
            </DialogFooter>
          </TabsContent>

          {/* Domain Context Tab */}
          <TabsContent value="context" className="space-y-4 py-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-2">
                  <FileJson className="h-4 w-4" />
                  Domain Context
                </Label>
                <div className="flex gap-2">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept=".json"
                    className="hidden"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="h-4 w-4 mr-1" />
                    Upload JSON
                  </Button>
                  {hasContext && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={clearDomainContext}
                      className="text-red-500 hover:text-red-600"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Define company-specific context (prospects, products, terminology) to generate more relevant test data
              </p>
            </div>

            {isLoadingContext ? (
              <div className="flex items-center justify-center h-48">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              </div>
            ) : (
              <>
                <div className="space-y-2">
                  <textarea
                    value={contextJson}
                    onChange={(e) => handleContextJsonChange(e.target.value)}
                    className="w-full h-64 font-mono text-xs p-3 rounded-md border bg-muted text-foreground resize-none"
                    placeholder={`{
  "company": {
    "name": "Your Company",
    "industry": "Technology"
  },
  "prospects": ["Company A", "Company B"],
  "people": ["John Smith", "Jane Doe"],
  "products": ["Product X", "Product Y"],
  "sample_queries": [
    "Summarize my last call with Company A",
    "What were the action items from the QBR?"
  ],
  "domain_terminology": {
    "QBR": "Quarterly Business Review",
    "ARR": "Annual Recurring Revenue"
  },
  "customer_types": ["Enterprise", "SMB"]
}`}
                  />
                  {contextError && (
                    <div className="flex items-center gap-1 text-red-500 text-xs">
                      <AlertCircle className="h-3 w-3" />
                      {contextError}
                    </div>
                  )}
                </div>

                {hasContext && (
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-md border border-green-200 dark:border-green-800">
                    <p className="text-xs text-green-700 dark:text-green-400 flex items-center gap-1">
                      <Check className="h-3 w-3" />
                      Domain context is active. Test data will use your company-specific information.
                    </p>
                  </div>
                )}

                <DialogFooter className="pt-4">
                  <Button variant="outline" onClick={() => onOpenChange(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={saveDomainContext}
                    disabled={isSavingContext || !!contextError}
                  >
                    {isSavingContext ? "Saving..." : "Save Context"}
                  </Button>
                </DialogFooter>
              </>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};

export default SettingsModal;
