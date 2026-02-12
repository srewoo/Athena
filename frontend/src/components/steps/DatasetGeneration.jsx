import { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { toast } from 'sonner';
import { Database, Loader2, Target, BarChart3, AlertTriangle, Plus } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function DatasetGeneration({ project, selectedVersion, settings }) {
  const [systemPrompt, setSystemPrompt] = useState('');
  const [sampleCount, setSampleCount] = useState(10);
  const [testCases, setTestCases] = useState([]);
  const [loading, setLoading] = useState(false);

  // Dimension-aware state
  const [generationMode, setGenerationMode] = useState('standard');
  const [evalDimensions, setEvalDimensions] = useState([]);
  const [casesPerDimension, setCasesPerDimension] = useState(5);
  const [dimensionLoading, setDimensionLoading] = useState(false);

  // Coverage state
  const [coverage, setCoverage] = useState(null);

  useEffect(() => {
    if (project) {
      loadTestCases();
      loadEvalDimensions();
    }
    if (selectedVersion) {
      setSystemPrompt(selectedVersion.content);
    } else {
      loadLatestVersion();
    }
  }, [project, selectedVersion]);

  // Load coverage whenever test cases change
  useEffect(() => {
    if (project && testCases.length > 0) {
      loadCoverage();
    }
  }, [testCases.length]);

  const loadLatestVersion = async () => {
    if (!project) return;
    try {
      const response = await axios.get(`${API}/prompt-versions/${project.id}`);
      if (response.data.length > 0) {
        setSystemPrompt(response.data[0].content);
      }
    } catch (error) {
      console.error('Error loading prompt versions:', error);
    }
  };

  const loadTestCases = async () => {
    try {
      const response = await axios.get(`${API}/test-cases/${project.id}`);
      setTestCases(response.data);
    } catch (error) {
      console.error('Error loading test cases:', error);
    }
  };

  const loadEvalDimensions = async () => {
    if (!project) return;
    try {
      const response = await axios.get(`${API}/eval-prompts/${project.id}`);
      const dims = response.data.map(ep => ({
        name: ep.dimension,
        description: ep.content?.substring(0, 300) || ''
      }));
      setEvalDimensions(dims);
    } catch (error) {
      console.error('Error loading eval dimensions:', error);
    }
  };

  const loadCoverage = async () => {
    if (!project) return;
    try {
      const response = await axios.get(`${API}/dimension-coverage/${project.id}`);
      setCoverage(response.data);
    } catch (error) {
      console.error('Error loading coverage:', error);
    }
  };

  const getApiKey = () => {
    const provider = settings?.default_provider || 'openai';
    if (provider === 'openai') return settings?.openai_key;
    if (provider === 'anthropic') return settings?.claude_key;
    if (provider === 'google') return settings?.gemini_key;
    return null;
  };

  // Standard generation (existing)
  const handleGenerate = async () => {
    const provider = settings?.default_provider || 'openai';
    const apiKey = getApiKey();

    if (!systemPrompt) { toast.error('System prompt is required'); return; }
    if (!apiKey) { toast.error(`Please configure ${provider.toUpperCase()} API key in Settings`); return; }
    if (sampleCount < 1 || sampleCount > 100) { toast.error('Sample count must be between 1 and 100'); return; }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/test-cases`, {
        project_id: project.id,
        prompt_version_id: selectedVersion?.id || 'latest',
        system_prompt: systemPrompt,
        sample_count: sampleCount,
        provider: provider,
        model: settings?.default_model || 'gpt-4o',
        api_key: apiKey,
      });

      setTestCases([...response.data, ...testCases]);
      toast.success(`Generated ${response.data.length} test cases!`);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Generation failed';
      toast.error(`Failed to generate test cases: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  // Dimension-aware generation (new)
  const handleDimensionAwareGenerate = async () => {
    const provider = settings?.default_provider || 'openai';
    const apiKey = getApiKey();

    if (!systemPrompt) { toast.error('System prompt is required'); return; }
    if (!apiKey) { toast.error(`Please configure ${provider.toUpperCase()} API key in Settings`); return; }
    if (evalDimensions.length === 0) { toast.error('No eval dimensions found. Generate eval prompts first (Step 3).'); return; }

    setDimensionLoading(true);
    try {
      const response = await axios.post(`${API}/generate-dimension-tests`, {
        project_id: project.id,
        prompt_version_id: selectedVersion?.id || 'latest',
        system_prompt: systemPrompt,
        dimensions: evalDimensions,
        cases_per_dimension: casesPerDimension,
        provider: provider,
        model: settings?.generation_model || settings?.default_model || 'gpt-4o',
        api_key: apiKey,
      });

      setTestCases([...response.data, ...testCases]);
      toast.success(`Generated ${response.data.length} dimension-targeted test cases across ${evalDimensions.length} dimensions!`);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Generation failed';
      toast.error(`Failed to generate dimension tests: ${errorMsg}`);
    } finally {
      setDimensionLoading(false);
    }
  };

  // Generate more tests for a specific under-covered dimension
  const handleGenerateForDimension = async (dimName) => {
    const provider = settings?.default_provider || 'openai';
    const apiKey = getApiKey();
    if (!apiKey) { toast.error('API key required'); return; }

    const dim = evalDimensions.find(d => d.name === dimName);
    if (!dim) return;

    setDimensionLoading(true);
    try {
      const response = await axios.post(`${API}/generate-dimension-tests`, {
        project_id: project.id,
        prompt_version_id: selectedVersion?.id || 'latest',
        system_prompt: systemPrompt,
        dimensions: [dim],
        cases_per_dimension: 5,
        provider: provider,
        model: settings?.generation_model || settings?.default_model || 'gpt-4o',
        api_key: apiKey,
      });

      setTestCases([...response.data, ...testCases]);
      toast.success(`Generated ${response.data.length} cases for "${dimName}"`);
    } catch (error) {
      toast.error(`Failed to generate for "${dimName}"`);
    } finally {
      setDimensionLoading(false);
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'positive': return 'default';
      case 'edge': return 'secondary';
      case 'negative': return 'destructive';
      case 'adversarial': return 'outline';
      default: return 'default';
    }
  };

  const getCoverageColor = (count) => {
    if (count >= 3) return 'text-green-600 bg-green-100 dark:bg-green-900/30';
    if (count >= 1) return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30';
    return 'text-red-600 bg-red-100 dark:bg-red-900/30';
  };

  const distribution = {
    positive: Math.round(sampleCount * 0.6),
    edge: Math.round(sampleCount * 0.2),
    negative: Math.round(sampleCount * 0.1),
    adversarial: Math.round(sampleCount * 0.1),
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
        <h2 className="text-3xl font-bold mb-2" data-testid="dataset-title">Step 4: Test Dataset Generation</h2>
        <p className="text-muted-foreground">
          AI-generated test cases with strategic distribution
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Generation Config */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Generate Test Cases</CardTitle>
            <CardDescription>Choose generation strategy</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Mode Toggle */}
            <div className="flex rounded-lg border border-border overflow-hidden">
              <button
                onClick={() => setGenerationMode('standard')}
                className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                  generationMode === 'standard'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-background hover:bg-muted'
                }`}
              >
                Standard
              </button>
              <button
                onClick={() => setGenerationMode('dimension')}
                className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                  generationMode === 'dimension'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-background hover:bg-muted'
                }`}
              >
                Dimension-Aware
              </button>
            </div>

            {generationMode === 'standard' ? (
              <>
                {/* Standard Mode */}
                <div className="space-y-2">
                  <Label htmlFor="sampleCount">Sample Count</Label>
                  <Input
                    id="sampleCount"
                    type="number"
                    min="1"
                    max="100"
                    value={sampleCount}
                    onChange={(e) => {
                      const value = parseInt(e.target.value);
                      if (!isNaN(value)) {
                        setSampleCount(Math.max(1, Math.min(100, value)));
                      }
                    }}
                    data-testid="sample-count-input"
                  />
                  <p className="text-xs text-muted-foreground">Generate between 1-100 test cases</p>
                </div>

                <div className="p-4 bg-muted rounded-md space-y-2 text-sm">
                  <p className="font-medium">Distribution:</p>
                  <div className="space-y-1 text-muted-foreground">
                    <div className="flex justify-between">
                      <span>Positive/Typical:</span>
                      <span className="font-medium">{distribution.positive} (60%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Edge Cases:</span>
                      <span className="font-medium">{distribution.edge} (20%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Negative:</span>
                      <span className="font-medium">{distribution.negative} (10%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Adversarial:</span>
                      <span className="font-medium">{distribution.adversarial} (10%)</span>
                    </div>
                  </div>
                </div>

                <Button
                  onClick={handleGenerate}
                  disabled={loading}
                  className="w-full button-hover"
                  data-testid="generate-tests-button"
                >
                  {loading ? (
                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Generating...</>
                  ) : (
                    <><Database className="w-4 h-4 mr-2" />Generate Test Cases</>
                  )}
                </Button>
              </>
            ) : (
              <>
                {/* Dimension-Aware Mode */}
                {evalDimensions.length === 0 ? (
                  <div className="p-4 bg-muted rounded-md text-sm text-muted-foreground">
                    <AlertTriangle className="w-4 h-4 inline mr-1" />
                    No eval dimensions found. Complete Step 3 (Eval Generation) first.
                  </div>
                ) : (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="casesPerDim">Cases per Dimension</Label>
                      <Input
                        id="casesPerDim"
                        type="number"
                        min="1"
                        max="20"
                        value={casesPerDimension}
                        onChange={(e) => {
                          const value = parseInt(e.target.value);
                          if (!isNaN(value)) {
                            setCasesPerDimension(Math.max(1, Math.min(20, value)));
                          }
                        }}
                      />
                    </div>

                    <div className="p-4 bg-muted rounded-md space-y-2 text-sm">
                      <p className="font-medium">Targeting {evalDimensions.length} dimensions:</p>
                      <div className="space-y-1 text-muted-foreground max-h-40 overflow-y-auto">
                        {evalDimensions.map(dim => (
                          <div key={dim.name} className="flex justify-between items-center">
                            <span className="truncate mr-2">{dim.name}</span>
                            <span className="font-medium text-xs whitespace-nowrap">{casesPerDimension} cases</span>
                          </div>
                        ))}
                      </div>
                      <div className="pt-2 border-t border-border">
                        <div className="flex justify-between font-medium">
                          <span>Total:</span>
                          <span>{evalDimensions.length * casesPerDimension} cases</span>
                        </div>
                      </div>
                    </div>

                    <Button
                      onClick={handleDimensionAwareGenerate}
                      disabled={dimensionLoading}
                      className="w-full button-hover"
                    >
                      {dimensionLoading ? (
                        <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Generating...</>
                      ) : (
                        <><Target className="w-4 h-4 mr-2" />Generate Dimension Tests</>
                      )}
                    </Button>
                  </>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Test Cases List */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Generated Test Cases ({testCases.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {testCases.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="font-medium mb-1">No test cases yet</p>
                <p className="text-sm">Click "Generate" to create test cases</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {testCases.map((tc, idx) => (
                  <div
                    key={tc.id}
                    className="p-4 border border-border rounded-md space-y-2 hover:border-primary/50 transition-colors"
                    data-testid={`test-case-${idx}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex gap-2">
                        <Badge variant={getTypeColor(tc.case_type)}>
                          {tc.case_type}
                        </Badge>
                        {tc.target_dimension && (
                          <Badge variant="secondary" className="text-xs">
                            <Target className="w-3 h-3 mr-1" />
                            {tc.target_dimension}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground font-medium mb-1">Input:</p>
                      <p className="text-sm font-mono bg-muted p-2 rounded">{tc.input_text}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground font-medium mb-1">Expected Behavior:</p>
                      <p className="text-sm">{tc.expected_behavior}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Dimension Coverage Card */}
      {coverage && coverage.total_dimensions > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Dimension Coverage
            </CardTitle>
            <CardDescription>
              {coverage.covered_dimensions}/{coverage.total_dimensions} dimensions covered ({coverage.coverage_percentage}%)
              {coverage.untagged_test_cases > 0 && (
                <span className="ml-2 text-muted-foreground">
                  + {coverage.untagged_test_cases} untagged
                </span>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Coverage Progress Bar */}
            <div className="w-full bg-muted rounded-full h-3 mb-4">
              <div
                className="bg-primary h-3 rounded-full transition-all"
                style={{ width: `${Math.min(100, coverage.coverage_percentage)}%` }}
              />
            </div>

            {/* Per-dimension breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(coverage.dimension_details).map(([dim, counts]) => (
                <div
                  key={dim}
                  className={`p-3 rounded-md border ${getCoverageColor(counts.total)}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm truncate">{dim}</span>
                    <span className="font-bold text-sm">{counts.total}</span>
                  </div>
                  <div className="flex gap-1 text-xs">
                    {counts.positive > 0 && <span>+{counts.positive}</span>}
                    {counts.edge > 0 && <span>E{counts.edge}</span>}
                    {counts.negative > 0 && <span>-{counts.negative}</span>}
                    {counts.adversarial > 0 && <span>A{counts.adversarial}</span>}
                  </div>
                  {counts.total < 3 && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="w-full mt-2 h-7 text-xs"
                      disabled={dimensionLoading}
                      onClick={() => handleGenerateForDimension(dim)}
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Generate More
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
