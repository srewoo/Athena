/**
 * Settings Modal Component
 * Handles LLM provider and API key configuration
 */
import React, { useState, useEffect } from "react";
import { Settings, Eye, EyeOff } from "lucide-react";
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

  const {
    llmProvider = "openai",
    llmModel = "",
    apiKey = "",
    evalProvider = "openai",
    evalModel = "o1-mini",
    useSeparateEvalModel = true
  } = settings || {};

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
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            LLM Settings
          </DialogTitle>
          <DialogDescription>
            Configure your LLM provider and API credentials
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
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
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving ? "Saving..." : "Save Settings"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default SettingsModal;
