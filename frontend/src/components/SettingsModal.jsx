import { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { toast } from 'sonner';
import { Settings, Eye, EyeOff, FileJson, Upload, Trash2, CheckCircle2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function SettingsModal({ open, onClose, settings, sessionId, onSettingsUpdated }) {
  const [openaiKey, setOpenaiKey] = useState('');
  const [claudeKey, setClaudeKey] = useState('');
  const [geminiKey, setGeminiKey] = useState('');
  const [defaultProvider, setDefaultProvider] = useState('openai');
  const [defaultModel, setDefaultModel] = useState('gpt-4o');
  const [metaEvalProvider, setMetaEvalProvider] = useState('google');
  const [metaEvalModel, setMetaEvalModel] = useState('gemini-2.5-flash');
  const [useSeparateEvalModel, setUseSeparateEvalModel] = useState(true);
  const [evalModel, setEvalModel] = useState('o1-mini');
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showClaudeKey, setShowClaudeKey] = useState(false);
  const [showGeminiKey, setShowGeminiKey] = useState(false);
  const [domainContext, setDomainContext] = useState('');
  const [maximApiKey, setMaximApiKey] = useState('');
  const [maximWorkspaceId, setMaximWorkspaceId] = useState('');
  const [maximRepositoryId, setMaximRepositoryId] = useState('');
  const [maximBaseUrl, setMaximBaseUrl] = useState('https://app.getmaxim.ai');
  const [saving, setSaving] = useState(false);
  const [domainContextValid, setDomainContextValid] = useState(false);
  const [contextStats, setContextStats] = useState({ products: 0, prospects: 0, competencies: 0 });

  useEffect(() => {
    if (settings) {
      setOpenaiKey(settings.openai_key || '');
      setClaudeKey(settings.claude_key || '');
      setGeminiKey(settings.gemini_key || '');
      setDefaultProvider(settings.default_provider || 'openai');
      setDefaultModel(settings.default_model || 'gpt-4o');
      setMetaEvalProvider(settings.meta_eval_provider || 'google');
      setMetaEvalModel(settings.meta_eval_model || 'gemini-2.5-flash');
      // Load execution model settings
      if (settings.execution_model) {
        setEvalModel(settings.execution_model);
        setUseSeparateEvalModel(settings.execution_model !== settings.default_model);
      }
      setMaximApiKey(settings.maxim_api_key || '');
      setMaximWorkspaceId(settings.maxim_workspace_id || '');
      setMaximRepositoryId(settings.maxim_repository_id || '');

      // Load domain context as JSON string
      if (settings.domain_context) {
        setDomainContext(JSON.stringify(settings.domain_context, null, 2));
      }
    }
  }, [settings]);

  const handleSave = async () => {
    setSaving(true);
    try {
      // Parse domain context JSON if provided
      let parsedDomainContext = undefined;
      if (domainContext && domainContext.trim()) {
        try {
          parsedDomainContext = JSON.parse(domainContext);
        } catch (e) {
          toast.error('Invalid domain context JSON', {
            description: 'Please check your JSON syntax'
          });
          setSaving(false);
          return;
        }
      }

      // Determine execution provider from the selected execution model
      const execModelInfo = getEvaluationModels().find(m => m.id === evalModel);
      const executionProvider = execModelInfo?.provider || 'openai';

      const response = await axios.post(`${API}/settings`, {
        session_id: sessionId,
        openai_key: openaiKey || undefined,
        claude_key: claudeKey || undefined,
        gemini_key: geminiKey || undefined,
        default_provider: defaultProvider,
        default_model: defaultModel,
        // Multi-model architecture: generation uses the default provider/model
        generation_provider: defaultProvider,
        generation_model: defaultModel,
        meta_eval_provider: metaEvalProvider,
        meta_eval_model: metaEvalModel,
        // Execution model for running evaluations
        execution_provider: useSeparateEvalModel ? executionProvider : defaultProvider,
        execution_model: useSeparateEvalModel ? evalModel : defaultModel,
        maxim_api_key: maximApiKey || undefined,
        maxim_workspace_id: maximWorkspaceId || undefined,
        maxim_repository_id: maximRepositoryId || undefined,
        domain_context: parsedDomainContext,
      });

      onSettingsUpdated(response.data);
      toast.success('Settings saved!');
    } catch (error) {
      console.error('Error saving settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const getProviderModels = (provider) => {
    const models = {
      openai: [
        { id: 'gpt-4o', name: 'GPT-4o' },
        { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
        { id: 'o1', name: 'O1' },
        { id: 'o1-mini', name: 'O1 Mini' },
        { id: 'o3', name: 'O3' },
        { id: 'o3-mini', name: 'O3 Mini' }
      ],
      anthropic: [
        { id: 'claude-sonnet-4.5', name: 'Claude Sonnet 4.5' },
        { id: 'claude-opus-4.5', name: 'Claude Opus 4.5' }
      ],
      google: [
        { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro' },
        { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash' },
        { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
        { id: 'gemini-2.0-flash-thinking-exp', name: 'Gemini 2.0 Flash Thinking (Exp)' }
      ],
    };
    return models[provider] || [];
  };

  const getMetaEvalModels = () => {
    return [
      // Fast models (recommended for high-volume)
      { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', provider: 'google', category: 'Fast' },
      { id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openai', category: 'Fast' },

      // Balanced models
      { id: 'gpt-4o', name: 'GPT-4o', provider: 'openai', category: 'Balanced' },
      { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', provider: 'google', category: 'Balanced' },
      { id: 'claude-sonnet-4.5', name: 'Claude Sonnet 4.5', provider: 'anthropic', category: 'Balanced' },

      // Reasoning/Thinking models (best for complex validation)
      { id: 'o1', name: 'O1 (Reasoning)', provider: 'openai', category: 'Thinking' },
      { id: 'o1-mini', name: 'O1 Mini (Reasoning)', provider: 'openai', category: 'Thinking' },
      { id: 'o3', name: 'O3 (Advanced Reasoning)', provider: 'openai', category: 'Thinking' },
      { id: 'o3-mini', name: 'O3 Mini (Reasoning)', provider: 'openai', category: 'Thinking' },
      { id: 'gemini-2.0-flash-thinking-exp', name: 'Gemini 2.0 Flash Thinking (Exp)', provider: 'google', category: 'Thinking' },

      // Premium models
      { id: 'claude-opus-4.5', name: 'Claude Opus 4.5', provider: 'anthropic', category: 'Premium' },
    ];
  };

  const getEvaluationModels = () => {
    return [
      { id: 'o1-mini', name: 'O1 Mini', provider: 'openai' },
      { id: 'o3-mini', name: 'O3 Mini', provider: 'openai' },
      { id: 'gpt-4o', name: 'GPT-4o', provider: 'openai' },
      { id: 'claude-sonnet-4.5', name: 'Claude Sonnet 4.5', provider: 'anthropic' },
    ];
  };

  const validateDomainContext = (jsonString) => {
    if (!jsonString || !jsonString.trim()) {
      setDomainContextValid(false);
      setContextStats({ products: 0, prospects: 0, competencies: 0 });
      return;
    }

    try {
      const parsed = JSON.parse(jsonString);
      setDomainContextValid(true);

      // Calculate stats
      const products = parsed.products?.length || 0;
      const prospects = parsed.prospects?.length || 0;
      const competencies = Object.keys(parsed.competency_taxonomy || {}).length;

      setContextStats({ products, prospects, competencies });
    } catch (e) {
      setDomainContextValid(false);
      setContextStats({ products: 0, prospects: 0, competencies: 0 });
    }
  };

  const handleDomainContextChange = (value) => {
    setDomainContext(value);
    validateDomainContext(value);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5" />
            <span>Settings</span>
          </DialogTitle>
          <DialogDescription>
            Configure LLM provider and domain context for test data generation
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="llm" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="llm">LLM Settings</TabsTrigger>
            <TabsTrigger value="domain">Domain Context</TabsTrigger>
            <TabsTrigger value="maxim">Maxim</TabsTrigger>
          </TabsList>

          {/* LLM Settings Tab */}
          <TabsContent value="llm" className="space-y-6 mt-4">
            {/* LLM Provider */}
            <div className="space-y-2">
              <Label htmlFor="llmProvider">LLM Provider</Label>
              <select
                id="llmProvider"
                value={defaultProvider}
                onChange={(e) => {
                  const provider = e.target.value;
                  setDefaultProvider(provider);
                  const models = getProviderModels(provider);
                  if (models.length > 0) setDefaultModel(models[0].id);
                }}
                className="w-full p-3 border border-input rounded-md bg-background text-base"
                data-testid="llm-provider-select"
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="google">Google</option>
              </select>
            </div>

            {/* Model */}
            <div className="space-y-2">
              <Label htmlFor="model">Model</Label>
              <select
                id="model"
                value={defaultModel}
                onChange={(e) => setDefaultModel(e.target.value)}
                className="w-full p-3 border border-input rounded-md bg-background text-base"
                data-testid="model-select"
              >
                {getProviderModels(defaultProvider).map(model => (
                  <option key={model.id} value={model.id}>{model.name}</option>
                ))}
              </select>
            </div>

            {/* API Key */}
            <div className="space-y-2">
              <Label htmlFor="apiKey">API Key</Label>
              <div className="relative">
                <Input
                  id="apiKey"
                  type={
                    defaultProvider === 'openai' ? (showOpenaiKey ? 'text' : 'password') :
                    defaultProvider === 'anthropic' ? (showClaudeKey ? 'text' : 'password') :
                    (showGeminiKey ? 'text' : 'password')
                  }
                  placeholder={
                    defaultProvider === 'openai' ? 'sk-...' :
                    defaultProvider === 'anthropic' ? 'sk-ant-...' :
                    'AIza...'
                  }
                  value={
                    defaultProvider === 'openai' ? openaiKey :
                    defaultProvider === 'anthropic' ? claudeKey :
                    geminiKey
                  }
                  onChange={(e) => {
                    if (defaultProvider === 'openai') setOpenaiKey(e.target.value);
                    else if (defaultProvider === 'anthropic') setClaudeKey(e.target.value);
                    else setGeminiKey(e.target.value);
                  }}
                  className="pr-10"
                  data-testid="api-key-input"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (defaultProvider === 'openai') setShowOpenaiKey(!showOpenaiKey);
                    else if (defaultProvider === 'anthropic') setShowClaudeKey(!showClaudeKey);
                    else setShowGeminiKey(!showGeminiKey);
                  }}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {(defaultProvider === 'openai' ? showOpenaiKey : 
                    defaultProvider === 'anthropic' ? showClaudeKey : 
                    showGeminiKey) ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
              <p className="text-xs text-muted-foreground">
                Your API key is stored locally and never sent to our servers
              </p>
            </div>

            {/* Meta-Evaluation Model Section */}
            <div className="border-t pt-6 space-y-4">
              <div className="space-y-1">
                <h3 className="text-sm font-semibold">Meta-Evaluation Settings</h3>
                <p className="text-xs text-muted-foreground">
                  Model used to validate eval quality (independent from generation model)
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="metaEvalModel">Meta-Eval Model</Label>
                <select
                  id="metaEvalModel"
                  value={metaEvalModel}
                  onChange={(e) => {
                    const selectedModel = e.target.value;
                    setMetaEvalModel(selectedModel);
                    // Auto-set provider based on selected model
                    const modelInfo = getMetaEvalModels().find(m => m.id === selectedModel);
                    if (modelInfo) {
                      setMetaEvalProvider(modelInfo.provider);
                    }
                  }}
                  className="w-full p-3 border border-input rounded-md bg-background text-base"
                >
                  <optgroup label="⚡ Fast (Recommended)">
                    {getMetaEvalModels().filter(m => m.category === 'Fast').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="⚖️ Balanced">
                    {getMetaEvalModels().filter(m => m.category === 'Balanced').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="🧠 Reasoning/Thinking">
                    {getMetaEvalModels().filter(m => m.category === 'Thinking').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="💎 Premium">
                    {getMetaEvalModels().filter(m => m.category === 'Premium').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                </select>
                <p className="text-xs text-muted-foreground">
                  💡 Fast models for high-volume, Thinking models for complex validation
                </p>
              </div>
            </div>

            {/* Separator */}
            <div className="border-t pt-6">
              {/* Use Separate Evaluation Model */}
              <div className="flex items-start space-x-3 mb-4">
                <input
                  type="checkbox"
                  id="useSeparateEval"
                  checked={useSeparateEvalModel}
                  onChange={(e) => setUseSeparateEvalModel(e.target.checked)}
                  className="mt-1 w-4 h-4 rounded border-gray-300"
                />
                <div className="flex-1">
                  <Label htmlFor="useSeparateEval" className="font-medium cursor-pointer">
                    Use separate model for evaluation
                  </Label>
                  <p className="text-xs text-muted-foreground mt-1">
                    Recommended: Use a reasoning model for more accurate evaluations
                  </p>
                </div>
              </div>

              {/* Evaluation Model */}
              {useSeparateEvalModel && (
                <div className="space-y-2 ml-7">
                  <Label htmlFor="evalModel">Evaluation Model</Label>
                  <select
                    id="evalModel"
                    value={evalModel}
                    onChange={(e) => setEvalModel(e.target.value)}
                    className="w-full p-3 border border-input rounded-md bg-background text-base"
                  >
                    {getEvaluationModels().map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Domain Context Tab */}
          <TabsContent value="domain" className="space-y-6 mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileJson className="w-5 h-5 text-muted-foreground" />
                  <Label className="text-base font-medium">Domain Context JSON</Label>
                </div>
                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const input = document.createElement('input');
                      input.type = 'file';
                      input.accept = '.json';
                      input.onchange = (e) => {
                        const file = e.target.files[0];
                        const reader = new FileReader();
                        reader.onload = (e) => {
                          handleDomainContextChange(e.target.result);
                          toast.success('JSON file loaded');
                        };
                        reader.readAsText(file);
                      };
                      input.click();
                    }}
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Upload JSON
                  </Button>
                  {domainContext && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setDomainContext('');
                        setDomainContextValid(false);
                        setContextStats({ products: 0, prospects: 0, competencies: 0 });
                        toast.success('Domain context cleared');
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>

              <p className="text-sm text-muted-foreground">
                Define company-specific context (prospects, products, quality principles, competencies) to generate domain-specific evaluation prompts and test data
              </p>

              <Textarea
                placeholder={`{\n  "prospects": ["CyberController", "DHI", "Firstsource"],\n  "products": ["Call AI", "LMS", "Analytics"],\n  "quality_principles": [\n    "Ground all observations in specific evidence",\n    "Score relative to competency framework"\n  ]\n}`}
                value={domainContext}
                onChange={(e) => handleDomainContextChange(e.target.value)}
                rows={14}
                className="font-mono text-sm"
              />

              {domainContextValid && (
                <div className="p-3 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-900 rounded-lg">
                  <div className="flex items-start space-x-2 mb-2">
                    <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-green-800 dark:text-green-200">
                        Domain context is valid and active
                      </p>
                      <p className="text-xs text-green-700 dark:text-green-300 mt-1">
                        {contextStats.products} products • {contextStats.prospects} prospects • {contextStats.competencies} competency categories
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {!domainContextValid && domainContext && (
                <div className="p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900 rounded-lg flex items-start space-x-2">
                  <span className="text-red-600 dark:text-red-400">⚠️</span>
                  <p className="text-sm text-red-800 dark:text-red-200">
                    Invalid JSON syntax. Please check your formatting.
                  </p>
                </div>
              )}

              <p className="text-xs text-muted-foreground">
                💡 Tip: Your domain context will be used to generate domain-specific evaluation prompts with your quality principles, anti-patterns, and competency framework embedded.
              </p>
            </div>
          </TabsContent>

          {/* Maxim Integration Tab - Placeholder */}
          <TabsContent value="maxim" className="space-y-6 mt-4">
            <div className="space-y-4">
              <div>
                <h3 className="text-base font-medium mb-2">Maxim Integration</h3>
                <p className="text-sm text-muted-foreground">
                  Configure Maxim LLMOps platform to sync your evaluation prompts as AI evaluators
                </p>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="maximApiKey">Maxim API Key</Label>
                  <Input
                    id="maximApiKey"
                    type="password"
                    placeholder="••••••••••••••"
                    value={maximApiKey}
                    onChange={(e) => setMaximApiKey(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Find your API key in Maxim settings
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="maximWorkspace">Workspace ID</Label>
                  <Input
                    id="maximWorkspace"
                    placeholder="clyzhb2lk00096keq04xxrvov"
                    value={maximWorkspaceId}
                    onChange={(e) => setMaximWorkspaceId(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Find your workspace ID in Maxim settings
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="maximRepository">Repository ID (Optional)</Label>
                  <Input
                    id="maximRepository"
                    placeholder="Enter your Maxim repository ID (optional)"
                    value={maximRepositoryId}
                    onChange={(e) => setMaximRepositoryId(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Optional: Used for advanced Maxim features like datasets
                  </p>
                </div>
              </div>

              {maximApiKey && maximWorkspaceId && (
                <div className="p-3 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-900 rounded-lg flex items-start space-x-2">
                  <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-green-800 dark:text-green-200">
                    Maxim integration is configured. You can now create evaluators in Maxim.
                  </p>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex justify-end space-x-2 mt-4">
          <Button variant="outline" onClick={onClose} data-testid="cancel-settings-button">
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving} data-testid="save-settings-button">
            {saving ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}