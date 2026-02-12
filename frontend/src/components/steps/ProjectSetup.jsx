import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Switch } from '../ui/switch';
import { Badge } from '../ui/badge';
import { toast } from 'sonner';
import { Sparkles, Loader2, Wand2, Plus, Trash2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function ProjectSetup({ onProjectCreated, sessionId, settings }) {
  const [name, setName] = useState('');
  const [useCase, setUseCase] = useState('');
  const [requirements, setRequirements] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const debounceTimerRef = useRef(null);

  // Multi-prompt support
  const [promptMode, setPromptMode] = useState('single'); // 'single' | 'multi'
  const [prompts, setPrompts] = useState([
    { id: 1, name: 'Main Prompt', content: '', order: 0 }
  ]);

  // Handle mode toggle
  const handleModeToggle = (checked) => {
    const newMode = checked ? 'multi' : 'single';
    setPromptMode(newMode);

    if (newMode === 'multi' && systemPrompt) {
      // Initialize first prompt with current systemPrompt
      setPrompts([{ id: 1, name: 'Main Prompt', content: systemPrompt, order: 0 }]);
    } else if (newMode === 'single' && prompts.length > 0) {
      // Concatenate all prompts when switching back to single mode
      const concatenated = prompts
        .sort((a, b) => a.order - b.order)
        .map((p, idx) => `=== PROMPT ${idx + 1}: ${p.name} ===\n${p.content}`)
        .join('\n\n');
      setSystemPrompt(concatenated);
    }
  };

  // Add new prompt in multi-mode
  const handleAddPrompt = () => {
    const newId = Math.max(...prompts.map(p => p.id), 0) + 1;
    setPrompts([...prompts, {
      id: newId,
      name: `Prompt ${prompts.length + 1}`,
      content: '',
      order: prompts.length
    }]);
  };

  // Remove prompt in multi-mode
  const handleRemovePrompt = (id) => {
    if (prompts.length === 1) {
      toast.error('Cannot remove the last prompt');
      return;
    }
    setPrompts(prompts.filter(p => p.id !== id));
  };

  // Update prompt content
  const handlePromptChange = (id, field, value) => {
    setPrompts(prompts.map(p =>
      p.id === id ? { ...p, [field]: value } : p
    ));
  };

  // Auto-extract use case and requirements from system prompt
  const extractFromPrompt = async (prompt) => {
    if (!prompt || prompt.trim().length < 50) {
      return; // Don't extract from very short prompts
    }

    setExtracting(true);
    try {
      // Use settings passed from parent component, or fetch if not available
      let currentSettings = settings;
      if (!currentSettings && sessionId) {
        const settingsRes = await axios.get(`${API}/settings/${sessionId}`);
        currentSettings = settingsRes.data;
      }

      if (!currentSettings || (!currentSettings.openai_key && !currentSettings.claude_key && !currentSettings.gemini_key)) {
        toast.error('Please configure your API keys in Settings first');
        setExtracting(false);
        return;
      }

      // Use the default provider and model from settings
      const provider = currentSettings.default_provider || 'openai';
      const model = currentSettings.default_model || 'gpt-4o';
      const apiKey = provider === 'openai' ? currentSettings.openai_key : 
                     provider === 'anthropic' ? currentSettings.claude_key : 
                     currentSettings.gemini_key;

      // Call dedicated extraction endpoint
      const response = await axios.post(`${API}/extract-project-info`, {
        prompt_content: prompt,
        provider: provider,
        model: model,
        api_key: apiKey
      });

      console.log('Extraction response:', response.data);
      
      if (response.data.success) {
        // Ensure we're setting string values
        const extractedUseCase = typeof response.data.use_case === 'string' 
          ? response.data.use_case 
          : JSON.stringify(response.data.use_case || '');
        
        const extractedRequirements = typeof response.data.requirements === 'string' 
          ? response.data.requirements 
          : JSON.stringify(response.data.requirements || '');
        
        console.log('Setting useCase:', extractedUseCase);
        console.log('Setting requirements:', extractedRequirements);
        
        setUseCase(extractedUseCase);
        setRequirements(extractedRequirements);
        
        toast.success('✨ Auto-extracted successfully!', {
          description: 'Review and edit the extracted information as needed',
          duration: 4000
        });
      } else {
        console.warn('Extraction unsuccessful:', response.data);
        toast.info('Could not auto-extract. Please fill manually.');
      }
    } catch (error) {
      console.error('Error extracting from prompt:', error);
      // Don't show error toast for extraction failures - it's an optional feature
      // Just log it and let user fill manually
      console.log('Auto-extraction failed, user can fill manually');
    } finally {
      setExtracting(false);
    }
  };

  // Debounce system prompt changes
  useEffect(() => {
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Determine prompt content based on mode
    let promptToExtract = '';
    if (promptMode === 'single') {
      promptToExtract = systemPrompt;
    } else {
      // Concatenate all prompts for extraction in multi-mode
      promptToExtract = prompts
        .filter(p => p.content && p.content.trim())
        .sort((a, b) => a.order - b.order)
        .map((p, idx) => `=== PROMPT ${idx + 1}: ${p.name} ===\n${p.content}`)
        .join('\n\n');
    }

    // Set new timer for 2 seconds
    if (promptToExtract && promptToExtract.trim().length > 50 && (settings || sessionId)) {
      debounceTimerRef.current = setTimeout(() => {
        extractFromPrompt(promptToExtract);
      }, 2000);
    }

    // Cleanup
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [systemPrompt, prompts, promptMode, settings, sessionId]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation based on mode
    if (!name || !useCase || !requirements) {
      toast.error('Please fill in all required fields');
      return;
    }

    if (promptMode === 'single' && !systemPrompt) {
      toast.error('Please enter a system prompt');
      return;
    }

    if (promptMode === 'multi' && prompts.every(p => !p.content || !p.content.trim())) {
      toast.error('Please fill in at least one prompt');
      return;
    }

    setLoading(true);
    try {
      // Prepare prompt content based on mode
      let promptContent;
      let promptData;
      let validPrompts = []; // Declare at function scope

      if (promptMode === 'single') {
        promptContent = systemPrompt;
        promptData = {
          mode: 'single',
          content: systemPrompt
        };
      } else {
        // Multi-mode: concatenate prompts with clear separators
        validPrompts = prompts.filter(p => p.content && p.content.trim());
        promptContent = validPrompts
          .sort((a, b) => a.order - b.order)
          .map((p, idx) => `=== PROMPT ${idx + 1}: ${p.name} ===\n${p.content}`)
          .join('\n\n');

        promptData = {
          mode: 'multi',
          prompts: validPrompts,
          concatenated: promptContent
        };
      }

      // Create project
      const projectRes = await axios.post(`${API}/projects`, {
        name,
        use_case: useCase,
        requirements,
        prompt_mode: promptMode
      });

      // Create initial prompt version with mode metadata
      await axios.post(`${API}/prompt-versions`, {
        project_id: projectRes.data.id,
        content: promptContent,
        prompt_mode: promptMode,
        prompt_data: promptData
      });

      toast.success('Project created successfully!', {
        description: promptMode === 'multi'
          ? `Created with ${validPrompts.length} prompts`
          : 'Single prompt mode'
      });
      onProjectCreated(projectRes.data);
    } catch (error) {
      console.error('Error creating project:', error);
      toast.error('Failed to create project');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto pb-24">
      <div className="mb-8">
        <h2 className="text-3xl font-bold mb-2" data-testid="project-setup-title">Step 1: Project Setup</h2>
        <p className="text-muted-foreground">Create a new project and define your initial system prompt</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-primary" />
            <span>Project Details</span>
          </CardTitle>
          <CardDescription>Define your project and initial prompt</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Project Name - First field */}
            <div className="space-y-2">
              <Label htmlFor="name">Project Name *</Label>
              <Input
                id="name"
                placeholder="e.g., Customer Support Bot"
                value={name}
                onChange={(e) => setName(e.target.value)}
                data-testid="project-name-input"
              />
            </div>

            {/* Prompt Mode Toggle */}
            <div className="space-y-3 p-4 border border-primary/20 rounded-lg bg-primary/5">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="prompt-mode" className="text-base font-semibold">
                    Prompt Mode
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    {promptMode === 'single'
                      ? 'Using single system prompt'
                      : 'Using multiple prompts (will be concatenated)'}
                  </p>
                </div>
                <div className="flex items-center space-x-3">
                  <span className={`text-sm ${promptMode === 'single' ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
                    Single
                  </span>
                  <Switch
                    id="prompt-mode"
                    checked={promptMode === 'multi'}
                    onCheckedChange={handleModeToggle}
                    data-testid="prompt-mode-toggle"
                  />
                  <span className={`text-sm ${promptMode === 'multi' ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
                    Multi
                  </span>
                </div>
              </div>
            </div>

            {/* Single Prompt Mode */}
            {promptMode === 'single' && (
              <div className="space-y-2">
                <Label htmlFor="systemPrompt" className="flex items-center justify-between">
                  <span>System Prompt *</span>
                  {extracting && (
                    <span className="flex items-center text-xs text-muted-foreground">
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      Auto-extracting...
                    </span>
                  )}
                </Label>
                <Textarea
                  id="systemPrompt"
                  placeholder="You are a helpful assistant that..."
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  rows={8}
                  className="font-mono text-sm"
                  data-testid="system-prompt-input"
                />
                <div className="flex items-start space-x-2 text-xs text-muted-foreground">
                  <Wand2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <p>
                    Tip: Type your system prompt and pause for 2 seconds. We'll automatically extract the use case and requirements!
                  </p>
                </div>
              </div>
            )}

            {/* Multi Prompt Mode */}
            {promptMode === 'multi' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="text-base">System Prompts *</Label>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={handleAddPrompt}
                    className="h-8"
                  >
                    <Plus className="w-4 h-4 mr-1" />
                    Add Prompt
                  </Button>
                </div>

                {prompts.map((prompt, index) => (
                  <div key={prompt.id} className="p-4 border rounded-lg space-y-3 bg-card">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary">{index + 1}</Badge>
                        <Input
                          placeholder="Prompt name"
                          value={prompt.name}
                          onChange={(e) => handlePromptChange(prompt.id, 'name', e.target.value)}
                          className="h-8 text-sm"
                        />
                      </div>
                      {prompts.length > 1 && (
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          onClick={() => handleRemovePrompt(prompt.id)}
                          className="h-8 text-destructive hover:text-destructive"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                    <Textarea
                      placeholder={`Enter prompt ${index + 1} content...`}
                      value={prompt.content}
                      onChange={(e) => handlePromptChange(prompt.id, 'content', e.target.value)}
                      rows={6}
                      className="font-mono text-sm"
                      data-testid={`multi-prompt-${index}`}
                    />
                  </div>
                ))}

                <div className="flex items-start space-x-2 text-xs text-muted-foreground p-3 bg-muted/50 rounded">
                  <Wand2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <p>
                    Tip: Prompts will be concatenated in order during execution. Pause for 2 seconds after editing to auto-extract use case and requirements!
                  </p>
                </div>
              </div>
            )}

            {/* Auto-extraction status indicator */}
            {extracting && (
              <div className="p-4 bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/30 rounded-lg animate-pulse">
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <Loader2 className="w-5 h-5 text-primary animate-spin" />
                    <Sparkles className="w-3 h-3 text-primary absolute -top-1 -right-1" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-primary">AI is analyzing your prompt...</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Extracting use case and comprehensive requirements
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            {/* Auto-extracted fields indicator */}
            {!extracting && (useCase || requirements) && (
              <div className="p-3 bg-primary/5 border border-primary/20 rounded-lg">
                <div className="flex items-center space-x-2 text-sm text-primary mb-1">
                  <Sparkles className="w-4 h-4" />
                  <span className="font-medium">Auto-extracted content</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Review and edit the extracted information below as needed
                </p>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="useCase" className="flex items-center space-x-2">
                <span>Use Case *</span>
                {extracting && (
                  <span className="flex items-center text-xs text-primary">
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                    Extracting...
                  </span>
                )}
              </Label>
              <Textarea
                id="useCase"
                placeholder={extracting ? "✨ Analyzing your prompt..." : "e.g., Answer customer questions about products"}
                value={useCase}
                onChange={(e) => setUseCase(e.target.value)}
                rows={3}
                data-testid="use-case-input"
                disabled={extracting}
                className={extracting ? 'border-primary/50 bg-primary/5 animate-pulse' : ''}
              />
            </div>

            <div className="space-y-2 relative">
              <Label htmlFor="requirements" className="flex items-center space-x-2">
                <span>Requirements *</span>
                {extracting && (
                  <span className="flex items-center text-xs text-primary">
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                    Extracting...
                  </span>
                )}
              </Label>
              <div className="relative">
                <Textarea
                  id="requirements"
                  placeholder={extracting ? "✨ Extracting comprehensive requirements..." : "List your requirements and constraints..."}
                  value={requirements}
                  onChange={(e) => setRequirements(e.target.value)}
                  rows={6}
                  data-testid="requirements-input"
                  disabled={extracting}
                  className={extracting ? 'border-primary/50 bg-primary/5 animate-pulse' : ''}
                />
              </div>
            </div>

            <Button
              type="submit"
              size="lg"
              disabled={loading || extracting}
              className="w-full button-hover"
              data-testid="create-project-button"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : extracting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Extracting...
                </>
              ) : (
                'Create Project & Continue'
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}