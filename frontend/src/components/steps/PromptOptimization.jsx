import { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { toast } from 'sonner';
import { Sparkles, TrendingUp, AlertCircle, CheckCircle2, Lightbulb, Loader2, Shield } from 'lucide-react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../ui/accordion';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function FormatAnalysisTab({ heuristic }) {
  const formatScore = heuristic.format_score || 0;
  const recommendations = heuristic.format_recommendations || [];

  return (
    <div className="space-y-4">
      {/* Format Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-muted/50 rounded-md">
          <div className="text-xs text-muted-foreground mb-1">Markdown Headers</div>
          <div className="text-2xl font-bold">{heuristic.markdown_headers || 0}</div>
        </div>
        <div className="p-3 bg-muted/50 rounded-md">
          <div className="text-xs text-muted-foreground mb-1">Bullet Points</div>
          <div className="text-2xl font-bold">{heuristic.bullet_points || 0}</div>
        </div>
        <div className="p-3 bg-muted/50 rounded-md">
          <div className="text-xs text-muted-foreground mb-1">XML Tags</div>
          <div className="text-2xl font-bold">{heuristic.xml_tags || 0}</div>
        </div>
        <div className="p-3 bg-muted/50 rounded-md">
          <div className="text-xs text-muted-foreground mb-1">Code Blocks</div>
          <div className="text-2xl font-bold">{heuristic.code_blocks || 0}</div>
        </div>
      </div>

      {/* Format Score */}
      <div className="p-4 bg-primary/10 border border-primary/30 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">Format Score</span>
          <span className="text-sm text-muted-foreground">
            For: {(heuristic.target_provider || 'openai').toUpperCase()}
          </span>
        </div>
        <div className="flex items-center space-x-3">
          <div className="text-3xl font-bold text-primary">
            {formatScore.toFixed(1)}/10
          </div>
          <div className="flex-1">
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${formatScore * 10}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Format Recommendations */}
      {recommendations.length > 0 && (
        <div className="space-y-2">
          <div className="text-sm font-medium flex items-center space-x-2">
            <Lightbulb className="w-4 h-4 text-primary" />
            <span>Format Recommendations</span>
          </div>
          {recommendations.map((rec, idx) => (
            <div key={idx} className="flex items-start space-x-2 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-md text-sm">
              <TrendingUp className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <p>{rec}</p>
            </div>
          ))}
        </div>
      )}

      {/* Structure Quality */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <QualityIndicator
          label="Clear Sections"
          isGood={heuristic.has_sections}
        />
        <QualityIndicator
          label="Uses Lists"
          isGood={heuristic.has_lists}
        />
        <QualityIndicator
          label="Good Spacing"
          isGood={heuristic.has_good_spacing}
        />
        <QualityIndicator
          label="Code Examples"
          isGood={heuristic.has_code_examples}
        />
      </div>
    </div>
  );
}

function QualityIndicator({ label, isGood }) {
  const bgColor = isGood ? 'bg-green-50 dark:bg-green-950/20' : 'bg-orange-50 dark:bg-orange-950/20';
  const Icon = isGood ? CheckCircle2 : AlertCircle;
  const iconColor = isGood ? 'text-green-600 dark:text-green-400' : 'text-orange-600 dark:text-orange-400';

  return (
    <div className={`flex items-center space-x-2 p-2 rounded-md ${bgColor}`}>
      <Icon className={`w-4 h-4 ${iconColor}`} />
      <span>{label}</span>
    </div>
  );
}

function ScorecardEntry({ label, score, statusColor }) {
  return (
    <div className={`p-2 rounded-md border text-xs ${statusColor}`}>
      <div className="flex items-center justify-between">
        <span className="font-medium capitalize">{label}</span>
        <span className="font-bold">{score}</span>
      </div>
    </div>
  );
}

function FindingCard({ finding, severityVariant }) {
  return (
    <div className="p-3 border rounded-md space-y-2">
      <div className="flex items-center gap-2">
        <Badge variant={severityVariant}>{finding.severity}</Badge>
        <span className="text-xs text-muted-foreground">{finding.location}</span>
      </div>
      <p className="text-sm">{finding.finding}</p>
      <div className="text-sm text-primary/80 bg-primary/5 p-2 rounded">
        <strong>Fix:</strong> {finding.suggestion}
      </div>
      {finding.suggested_addition && (
        <pre className="text-xs bg-muted p-2 rounded overflow-x-auto whitespace-pre-wrap font-mono max-h-48 overflow-y-auto">
          {finding.suggested_addition}
        </pre>
      )}
    </div>
  );
}

function DimensionItem({ dim, index, getStatusColor, getSeverityVariant }) {
  const findings = dim.findings || [];
  const statusColor = getStatusColor(dim.status);
  const scoreText = dim.score != null ? dim.score.toFixed(1) : '—';
  const findingCount = findings.length;
  const findingLabel = findingCount === 1 ? '1 finding' : `${findingCount} findings`;

  return (
    <AccordionItem value={`dim-${index}`} className="border rounded-lg px-3">
      <AccordionTrigger className="hover:no-underline">
        <div className="flex items-center gap-3 w-full">
          <span className="font-medium text-sm">{dim.name}</span>
          <Badge variant="outline" className={`ml-auto mr-2 ${statusColor}`}>
            {scoreText}
          </Badge>
          {findingCount > 0 && (
            <span className="text-xs text-muted-foreground">{findingLabel}</span>
          )}
        </div>
      </AccordionTrigger>
      <AccordionContent>
        <p className="text-sm text-muted-foreground mb-3">{dim.summary}</p>
        {findingCount > 0 ? (
          <div className="space-y-3">
            {findings.map((f, fi) => (
              <FindingCard key={fi} finding={f} severityVariant={getSeverityVariant(f.severity)} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-green-600">No issues found.</p>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}

function PriorityFixCard({ fix, effortColor }) {
  return (
    <div className="p-3 border rounded-md space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-bold bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center shrink-0">
          {fix.priority}
        </span>
        <span className="font-medium text-sm flex-1">{fix.title}</span>
        <Badge variant="outline" className={effortColor}>
          {fix.effort} effort
        </Badge>
      </div>
      <p className="text-sm text-muted-foreground">{fix.impact}</p>
      <p className="text-sm">{fix.details}</p>
      {fix.suggested_code && (
        <pre className="text-xs bg-muted p-2 rounded overflow-x-auto whitespace-pre-wrap font-mono max-h-48 overflow-y-auto">
          {fix.suggested_code}
        </pre>
      )}
    </div>
  );
}

function AuditTab({ audit }) {
  if (!audit) return null;

  const getStatusColor = (status) => {
    switch (status) {
      case 'pass': return 'bg-green-100 dark:bg-green-950/30 text-green-800 dark:text-green-300 border-green-300 dark:border-green-800';
      case 'needs_attention': return 'bg-yellow-100 dark:bg-yellow-950/30 text-yellow-800 dark:text-yellow-300 border-yellow-300 dark:border-yellow-800';
      case 'fail': return 'bg-red-100 dark:bg-red-950/30 text-red-800 dark:text-red-300 border-red-300 dark:border-red-800';
      default: return 'bg-muted';
    }
  };

  const getSeverityVariant = (severity) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  const getEffortColor = (effort) => {
    switch (effort) {
      case 'low': return 'bg-green-100 text-green-800 dark:bg-green-950/30 dark:text-green-300';
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-950/30 dark:text-yellow-300';
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-950/30 dark:text-red-300';
      default: return '';
    }
  };

  // Pre-compute data to avoid chained member expressions in JSX
  const overallScore = audit.overall_score != null ? audit.overall_score.toFixed(1) : '—';
  const promptType = (audit.prompt_type || 'other').replace(/_/g, ' ');
  const description = audit.what_this_prompt_does || '';
  const strengths = audit.strengths || [];
  const dimensions = audit.audit_dimensions || [];
  const fixes = audit.priority_fixes || [];

  const scorecardRaw = audit.scorecard || {};
  const scorecardItems = Object.keys(scorecardRaw).map((key) => {
    const entry = scorecardRaw[key];
    const label = key.replace(/_/g, ' ');
    const score = entry.score != null ? entry.score.toFixed(1) : '—';
    const statusColor = getStatusColor(entry.status);
    return { key, label, score, statusColor };
  });

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-sm text-muted-foreground">Overall Audit Score</div>
            <div className="text-4xl font-bold text-primary">{overallScore}/10</div>
          </div>
          <Badge variant="outline" className="capitalize">{promptType}</Badge>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>

      {/* Scorecard Grid */}
      <div>
        <h4 className="text-sm font-semibold mb-3">Scorecard</h4>
        <div className="grid grid-cols-2 gap-2">
          {scorecardItems.map((item) => (
            <ScorecardEntry key={item.key} label={item.label} score={item.score} statusColor={item.statusColor} />
          ))}
        </div>
      </div>

      {/* Strengths */}
      {strengths.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-green-600" /> Strengths
          </h4>
          <div className="space-y-1">
            {strengths.map((s, i) => (
              <div key={i} className="text-sm p-2 bg-green-50 dark:bg-green-950/20 rounded-md">{s}</div>
            ))}
          </div>
        </div>
      )}

      {/* Dimension Details */}
      <div>
        <h4 className="text-sm font-semibold mb-3">Detailed Findings</h4>
        <Accordion type="multiple" className="space-y-2">
          {dimensions.map((dim, idx) => (
            <DimensionItem key={idx} dim={dim} index={idx} getStatusColor={getStatusColor} getSeverityVariant={getSeverityVariant} />
          ))}
        </Accordion>
      </div>

      {/* Priority Fixes */}
      {fixes.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-orange-600" /> Priority Fixes
          </h4>
          <div className="space-y-3">
            {fixes.map((fix, i) => (
              <PriorityFixCard key={i} fix={fix} effortColor={getEffortColor(fix.effort)} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function PromptOptimization({ project, selectedVersion, settings, onVersionCreated }) {
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [rewriting, setRewriting] = useState(false);
  const [hasAutoAnalyzed, setHasAutoAnalyzed] = useState(false);
  const [targetProvider, setTargetProvider] = useState('openai');  // Format optimization target
  const [optimizingFormat, setOptimizingFormat] = useState(false);
  const [auditResult, setAuditResult] = useState(null);
  const [auditing, setAuditing] = useState(false);

  useEffect(() => {
    if (selectedVersion) {
      setCurrentPrompt(selectedVersion.content);
      if (selectedVersion.analysis_data) {
        setAnalysis(selectedVersion.analysis_data);
        setHasAutoAnalyzed(true); // Already has analysis
      } else {
        setHasAutoAnalyzed(false); // Needs analysis
      }
    } else {
      loadLatestVersion();
    }
  }, [project]);

  // Auto-trigger analysis when prompt loads without existing analysis
  useEffect(() => {
    const shouldAutoAnalyze = 
      currentPrompt && 
      !analysis && 
      !hasAutoAnalyzed && 
      !loading && 
      settings?.openai_key;

    if (shouldAutoAnalyze) {
      setHasAutoAnalyzed(true);
      toast.info('Auto-analyzing and auditing your prompt...', {
        duration: 2000
      });
      setTimeout(() => {
        handleAnalyze(true);
        handleDeepAudit();
      }, 500);
    }
  }, [currentPrompt, analysis, hasAutoAnalyzed, loading, settings]);

  const loadLatestVersion = async () => {
    if (!project) return;
    try {
      const response = await axios.get(`${API}/prompt-versions/${project.id}`);
      if (response.data.length > 0) {
        const latest = response.data[0];
        setCurrentPrompt(latest.content);
        if (latest.analysis_data) {
          setAnalysis(latest.analysis_data);
          setHasAutoAnalyzed(true);
        } else {
          setHasAutoAnalyzed(false);
        }
      }
    } catch (error) {
      console.error('Error loading prompt versions:', error);
    }
  };

  const handleAnalyze = async (isAutoTriggered = false) => {
    const provider = settings?.default_provider || 'openai';
    
    // Get the correct API key based on provider
    let apiKey;
    if (provider === 'openai') {
      apiKey = settings?.openai_key;
    } else if (provider === 'anthropic') {
      apiKey = settings?.claude_key;
    } else if (provider === 'google') {
      apiKey = settings?.gemini_key;
    }

    if (!currentPrompt) {
      toast.error('Please enter a prompt first');
      return;
    }

    if (!apiKey) {
      toast.error(`Please configure ${provider.toUpperCase()} API key in Settings`);
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/analyze`, {
        prompt_content: currentPrompt,
        provider: provider,
        model: settings.default_model || 'gpt-4o',
        api_key: apiKey,
        target_provider: targetProvider,  // Provider to optimize format for
      });

      setAnalysis(response.data);
      
      if (isAutoTriggered) {
        toast.success('✨ Analysis complete! Review the improvements below.', {
          description: `Quality Score: ${response.data.combined_score?.toFixed(1)}/10`,
          duration: 5000
        });
      } else {
        toast.success('Analysis complete!');
      }
    } catch (error) {
      console.error('Error analyzing prompt:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Analysis failed';
      toast.error(`Analysis failed: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoRewrite = async () => {
    if (!currentPrompt || !settings?.openai_key) {
      toast.error('Please configure API keys in Settings');
      return;
    }

    // Build comprehensive feedback with suggestions, issues, and strengths
    let comprehensiveFeedback = '';
    
    // Add current quality score
    if (analysis?.combined_score) {
      comprehensiveFeedback += `CURRENT QUALITY SCORE: ${analysis.combined_score.toFixed(1)}/10\n\n`;
    }
    
    // Add issues to fix
    if (analysis?.llm?.issues && analysis.llm.issues.length > 0) {
      comprehensiveFeedback += '🔴 ISSUES TO FIX:\n';
      analysis.llm.issues.forEach((issue, idx) => {
        comprehensiveFeedback += `${idx + 1}. ${issue}\n`;
      });
      comprehensiveFeedback += '\n';
    }
    
    // Add suggestions for improvement
    if (analysis?.llm?.suggestions && analysis.llm.suggestions.length > 0) {
      comprehensiveFeedback += '💡 IMPROVEMENTS TO APPLY:\n';
      analysis.llm.suggestions.forEach((sug, idx) => {
        comprehensiveFeedback += `${idx + 1}. [${sug.priority.toUpperCase()}] ${sug.suggestion}\n`;
      });
      comprehensiveFeedback += '\n';
    }
    
    // Add strengths to preserve
    if (analysis?.llm?.strengths && analysis.llm.strengths.length > 0) {
      comprehensiveFeedback += '✅ STRENGTHS TO PRESERVE:\n';
      analysis.llm.strengths.forEach((strength, idx) => {
        comprehensiveFeedback += `${idx + 1}. ${strength}\n`;
      });
      comprehensiveFeedback += '\n';
    }
    
    // Fallback if no analysis data
    if (!comprehensiveFeedback) {
      comprehensiveFeedback = 'Improve clarity, structure, effectiveness, and overall quality of the prompt.';
    }
    
    // Log the comprehensive feedback being sent
    console.log('=== COMPREHENSIVE FEEDBACK SENT TO LLM ===');
    console.log(comprehensiveFeedback);
    console.log('===========================================');

    // Get correct API key based on provider
    const provider = settings.default_provider || 'openai';
    let apiKey;
    if (provider === 'openai') {
      apiKey = settings.openai_key;
    } else if (provider === 'anthropic') {
      apiKey = settings.claude_key;
    } else if (provider === 'google') {
      apiKey = settings.gemini_key;
    }

    if (!apiKey) {
      toast.error(`Please configure ${provider.toUpperCase()} API key in Settings`);
      return;
    }

    setRewriting(true);
    try {
      const response = await axios.post(`${API}/rewrite`, {
        prompt_content: currentPrompt,
        feedback: comprehensiveFeedback,
        provider: provider,
        model: settings.default_model || 'gpt-4o',
        api_key: apiKey,
      });

      const rewritten = response.data.rewritten_prompt;
      setCurrentPrompt(rewritten);
      
      // Create new version
      const versionRes = await axios.post(`${API}/prompt-versions`, {
        project_id: project.id,
        content: rewritten,
      });

      onVersionCreated(versionRes.data);
      setAnalysis(null);
      setHasAutoAnalyzed(false); // Reset to allow auto-analyze of new version
      toast.info('🎯 Prompt rewritten! Analyzing improvements...', {
        duration: 2000
      });
      
      // Auto-trigger analysis + audit after rewrite to show improvements
      setTimeout(() => {
        handleAnalyze(true);
        handleDeepAudit();
        setHasAutoAnalyzed(true);
      }, 500);
    } catch (error) {
      console.error('Error rewriting prompt:', error);
      toast.error('Rewrite failed');
    } finally {
      setRewriting(false);
    }
  };

  const handleOptimizeFormat = async () => {
    if (!currentPrompt || !settings?.openai_key) {
      toast.error('Please configure API keys in Settings');
      return;
    }

    const provider = settings.default_provider || 'openai';
    let apiKey;
    if (provider === 'openai') {
      apiKey = settings.openai_key;
    } else if (provider === 'anthropic') {
      apiKey = settings.claude_key;
    } else if (provider === 'google') {
      apiKey = settings.gemini_key;
    }

    if (!apiKey) {
      toast.error(`Please configure ${provider.toUpperCase()} API key in Settings`);
      return;
    }

    setOptimizingFormat(true);
    try {
      const response = await axios.post(`${API}/optimize-format`, {
        prompt_content: currentPrompt,
        target_provider: targetProvider,
        provider: provider,
        model: settings.default_model || 'gpt-4o',
        api_key: apiKey,
      });

      const optimized = response.data.optimized_prompt;
      setCurrentPrompt(optimized);

      // Create new version
      const versionRes = await axios.post(`${API}/prompt-versions`, {
        project_id: project.id,
        content: optimized,
      });

      onVersionCreated(versionRes.data);
      setAnalysis(null);
      setHasAutoAnalyzed(false);
      toast.success(`✨ Format optimized for ${targetProvider.toUpperCase()}!`, {
        description: 'Analyzing improvements...',
        duration: 3000
      });

      // Auto-trigger analysis + audit after format optimization
      setTimeout(() => {
        handleAnalyze(true);
        handleDeepAudit();
        setHasAutoAnalyzed(true);
      }, 500);
    } catch (error) {
      console.error('Error optimizing format:', error);
      toast.error('Format optimization failed');
    } finally {
      setOptimizingFormat(false);
    }
  };

  const handleManualUpdate = async () => {
    if (!currentPrompt) return;

    try {
      const versionRes = await axios.post(`${API}/prompt-versions`, {
        project_id: project.id,
        content: currentPrompt,
      });

      onVersionCreated(versionRes.data);
      toast.success('New version saved!');
    } catch (error) {
      console.error('Error saving version:', error);
      toast.error('Failed to save version');
    }
  };

  const handleDeepAudit = async () => {
    const provider = settings?.default_provider || 'openai';
    let apiKey;
    if (provider === 'openai') apiKey = settings?.openai_key;
    else if (provider === 'anthropic') apiKey = settings?.claude_key;
    else if (provider === 'google') apiKey = settings?.gemini_key;

    if (!currentPrompt) {
      toast.error('Please enter a prompt first');
      return;
    }
    if (!apiKey) {
      toast.error(`Please configure ${provider.toUpperCase()} API key in Settings`);
      return;
    }

    setAuditing(true);
    try {
      const response = await axios.post(`${API}/audit-prompt`, {
        prompt_content: currentPrompt,
        provider: provider,
        model: settings.default_model || 'gpt-4o',
        api_key: apiKey,
        project_id: project?.id || null,
      });
      setAuditResult(response.data);
      toast.success(`Deep audit complete! Score: ${response.data.overall_score?.toFixed(1)}/10`);
    } catch (error) {
      console.error('Error auditing prompt:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Audit failed';
      toast.error(`Audit failed: ${errorMsg}`);
    } finally {
      setAuditing(false);
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'default';
    }
  };

  if (!project) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Please create a project first</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-24">
      <div>
        <h2 className="text-3xl font-bold mb-2" data-testid="optimization-title">Step 2: Prompt Optimization</h2>
        <p className="text-muted-foreground">Analyze and improve your system prompt with AI-powered insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prompt Editor */}
        <Card>
          <CardHeader>
            <CardTitle>System Prompt</CardTitle>
            <CardDescription>Edit and refine your prompt</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Target Provider Selector */}
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border border-border">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Optimize Format For:</span>
                </div>
                <select
                  value={targetProvider}
                  onChange={(e) => setTargetProvider(e.target.value)}
                  className="px-3 py-1.5 text-sm border border-input rounded-md bg-background"
                >
                  <option value="openai">OpenAI (Markdown)</option>
                  <option value="anthropic">Anthropic Claude (XML)</option>
                  <option value="google">Google Gemini (Structured)</option>
                </select>
              </div>
              <p className="text-xs text-muted-foreground px-1">
                Optimize Format adds structure (headers, bullets, etc.) without changing wording
              </p>
            </div>

            <Textarea
              value={currentPrompt}
              onChange={(e) => setCurrentPrompt(e.target.value)}
              rows={16}
              className="font-mono text-sm resize-none"
              data-testid="prompt-editor"
            />
            <div className="grid grid-cols-2 gap-2">
              <Button
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full"
                data-testid="analyze-button"
              >
                {loading ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Analyzing...</>
                ) : (
                  <><Sparkles className="w-4 h-4 mr-2" />Analyze</>
                )}
              </Button>
              <Button
                onClick={handleOptimizeFormat}
                disabled={optimizingFormat}
                variant="default"
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                data-testid="optimize-format-button"
                title="Add formatting (headers, bullets, etc.) without changing wording"
              >
                {optimizingFormat ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Optimizing...</>
                ) : (
                  <><TrendingUp className="w-4 h-4 mr-2" />Optimize Format</>
                )}
              </Button>
              <Button
                onClick={handleAutoRewrite}
                disabled={rewriting || !analysis}
                variant="outline"
                className="w-full"
                data-testid="auto-rewrite-button"
              >
                {rewriting ? 'Rewriting...' : 'Auto-Rewrite'}
              </Button>
              <Button
                onClick={handleManualUpdate}
                variant="secondary"
                className="w-full"
                data-testid="save-version-button"
              >
                Save Version
              </Button>
            </div>
            <Button
              onClick={handleDeepAudit}
              disabled={auditing || !currentPrompt}
              className="w-full bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white"
              data-testid="deep-audit-button"
            >
              {auditing ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Running Deep Audit...</>
              ) : (
                <><Shield className="w-4 h-4 mr-2" />Deep Audit</>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Analysis Results */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            {analysis && (
              <div className="flex items-center space-x-2 mt-2">
                <span className="text-sm text-muted-foreground">Quality Score:</span>
                <span className="text-2xl font-bold text-primary">
                  {analysis.combined_score?.toFixed(1)}/10
                </span>
                <TrendingUp className="w-5 h-5 text-primary" />
              </div>
            )}
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-12">
                <Loader2 className="w-12 h-12 mx-auto mb-4 text-primary animate-spin" />
                <p className="text-primary font-medium">Analyzing your prompt...</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Evaluating against industry best practices
                </p>
              </div>
            ) : !analysis ? (
              <div className="text-center py-12 text-muted-foreground">
                <Lightbulb className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Analysis will appear here</p>
                <p className="text-xs mt-2">Auto-analyzing on first load...</p>
              </div>
            ) : (
              <Tabs defaultValue={auditResult ? "audit" : "suggestions"} className="w-full">
                <TabsList className="grid w-full grid-cols-5">
                  <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
                  <TabsTrigger value="format">Format</TabsTrigger>
                  <TabsTrigger value="strengths">Strengths</TabsTrigger>
                  <TabsTrigger value="issues">Issues</TabsTrigger>
                  <TabsTrigger value="audit">
                    Audit {auditResult && <Badge variant="outline" className="ml-1 text-xs">{auditResult.overall_score?.toFixed(1)}</Badge>}
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="suggestions" className="space-y-3 mt-4">
                  {analysis.llm?.suggestions?.map((sug, idx) => (
                    <div key={idx} className="p-4 border border-border rounded-md space-y-2">
                      <div className="flex items-start justify-between">
                        <Badge variant={getPriorityColor(sug.priority)}>
                          {sug.priority}
                        </Badge>
                        <Badge variant="outline">{sug.category}</Badge>
                      </div>
                      <p className="text-sm">{sug.suggestion}</p>
                    </div>
                  ))}
                </TabsContent>

                <TabsContent value="format" className="space-y-4 mt-4">
                  {analysis?.heuristic ? (
                    <FormatAnalysisTab heuristic={analysis.heuristic} />
                  ) : (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      Run analysis to see format metrics
                    </p>
                  )}
                </TabsContent>

                <TabsContent value="strengths" className="space-y-2 mt-4">
                  {analysis.llm?.strengths?.map((strength, idx) => (
                    <div key={idx} className="flex items-start space-x-2 p-3 bg-green-50 dark:bg-green-950/20 rounded-md">
                      <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
                      <p className="text-sm">{strength}</p>
                    </div>
                  ))}
                </TabsContent>

                <TabsContent value="issues" className="space-y-2 mt-4">
                  {analysis.llm?.issues?.map((issue, idx) => (
                    <div key={idx} className="flex items-start space-x-2 p-3 bg-orange-50 dark:bg-orange-950/20 rounded-md">
                      <AlertCircle className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5 flex-shrink-0" />
                      <p className="text-sm">{issue}</p>
                    </div>
                  ))}
                </TabsContent>

                <TabsContent value="audit" className="space-y-4 mt-4">
                  {auditing ? (
                    <div className="text-center py-12">
                      <Loader2 className="w-12 h-12 mx-auto mb-4 text-orange-500 animate-spin" />
                      <p className="text-orange-600 font-medium">Running deep audit...</p>
                      <p className="text-xs text-muted-foreground mt-2">
                        Analyzing 9 dimensions across your prompt (30-60s)
                      </p>
                    </div>
                  ) : auditResult ? (
                    <AuditTab audit={auditResult} />
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <Shield className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>Click "Deep Audit" to run expert-level analysis</p>
                      <p className="text-xs mt-2">Evaluates 9 dimensions including intent alignment</p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}