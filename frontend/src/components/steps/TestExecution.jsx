import { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Label } from '../ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { toast } from 'sonner';
import { Play, ChevronDown, CheckCircle2, XCircle, Download, Loader2, Clock, History } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function TestExecution({ project, selectedVersion, settings }) {
  const [systemPrompt, setSystemPrompt] = useState('');
  const [testCases, setTestCases] = useState([]);
  const [evalPrompts, setEvalPrompts] = useState([]);
  const [selectedEval, setSelectedEval] = useState(null);
  const [selectedTests, setSelectedTests] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedRows, setExpandedRows] = useState({});
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [allResults, setAllResults] = useState([]);
  const [executionRuns, setExecutionRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);

  useEffect(() => {
    if (project) {
      loadTestCases();
      loadEvalPrompts();
      loadPastResults();
    }
    if (selectedVersion) {
      setSystemPrompt(selectedVersion.content);
    } else {
      loadLatestVersion();
    }
  }, [project, selectedVersion]);

  useEffect(() => {
    if (settings) {
      setSelectedProvider(settings.default_provider || 'openai');
      setSelectedModel(settings.execution_model || settings.default_model || 'gpt-4o');
    }
  }, [settings]);

  // Update model when provider changes
  useEffect(() => {
    if (selectedProvider === 'openai') {
      setSelectedModel('gpt-4o');
    } else if (selectedProvider === 'anthropic') {
      setSelectedModel('claude-sonnet-4-5-20250929');
    } else if (selectedProvider === 'google') {
      setSelectedModel('gemini-2.5-flash-preview-01-21');
    }
  }, [selectedProvider]);

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
      setSelectedTests(response.data.map(tc => tc.id));
    } catch (error) {
      console.error('Error loading test cases:', error);
    }
  };

  const loadEvalPrompts = async () => {
    try {
      const response = await axios.get(`${API}/eval-prompts/${project.id}`);
      setEvalPrompts(response.data);
      if (response.data.length > 0) {
        setSelectedEval(response.data[0]);
      }
    } catch (error) {
      console.error('Error loading eval prompts:', error);
    }
  };

  const loadPastResults = async () => {
    try {
      const response = await axios.get(`${API}/test-results/${project.id}`);
      const pastResults = response.data;
      setAllResults(pastResults);

      // Group results by execution run (prompt_version_id + eval_prompt_id + created_at proximity)
      const runs = {};
      pastResults.forEach(result => {
        // Group results within 5 minutes of each other as same run
        const runKey = `${result.prompt_version_id}-${result.eval_prompt_id}`;
        const timestamp = new Date(result.created_at).getTime();

        let foundRun = false;
        for (const key in runs) {
          const runTimestamp = new Date(runs[key][0].created_at).getTime();
          if (Math.abs(timestamp - runTimestamp) < 5 * 60 * 1000) { // 5 minutes
            runs[key].push(result);
            foundRun = true;
            break;
          }
        }

        if (!foundRun) {
          const uniqueKey = `${runKey}-${timestamp}`;
          runs[uniqueKey] = [result];
        }
      });

      // Convert to array and sort by most recent first
      const runsArray = Object.entries(runs).map(([key, results]) => ({
        id: key,
        results: results,
        timestamp: new Date(results[0].created_at),
        evalPromptId: results[0].eval_prompt_id,
        promptVersionId: results[0].prompt_version_id,
        totalTests: results.length,
        passed: results.filter(r => r.passed).length,
        avgScore: (results.reduce((sum, r) => sum + r.score, 0) / results.length).toFixed(1),
      })).sort((a, b) => b.timestamp - a.timestamp);

      setExecutionRuns(runsArray);

      // Load most recent results by default
      if (runsArray.length > 0) {
        setResults(runsArray[0].results);
        setSelectedRun(runsArray[0].id);
      }
    } catch (error) {
      console.error('Error loading past results:', error);
    }
  };

  const handleExecute = async () => {
    // Get the correct API key based on selected provider
    let apiKey;
    if (selectedProvider === 'openai') {
      apiKey = settings?.openai_key;
    } else if (selectedProvider === 'anthropic') {
      apiKey = settings?.claude_key;
    } else if (selectedProvider === 'google') {
      apiKey = settings?.gemini_key;
    }

    if (!selectedEval || selectedTests.length === 0 || !apiKey) {
      toast.error(`Please select eval prompt, test cases, and configure ${selectedProvider.toUpperCase()} API key in Settings`);
      return;
    }

    setLoading(true);
    setResults([]); // Clear previous results

    try {
      const response = await fetch(`${API}/execute-tests`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_id: project.id,
          prompt_version_id: selectedVersion?.id || 'latest',
          eval_prompt_id: selectedEval.id,
          system_prompt: systemPrompt,
          eval_prompt_content: selectedEval.content,
          test_case_ids: selectedTests,
          provider: selectedProvider,
          model: selectedModel,
          api_key: apiKey,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);

              if (data.type === 'start') {
                toast.info(`Starting execution of ${data.total} tests...`);
              } else if (data.type === 'heartbeat') {
                toast.info(`Processing test ${data.current}/${data.total}...`, { duration: 1000 });
              } else if (data.type === 'progress') {
                toast.info(data.message, { duration: 1000 });
              } else if (data.type === 'complete') {
                setResults(data.results || []);
                toast.success(`✅ Executed ${data.total} tests using ${selectedProvider.toUpperCase()} ${selectedModel}! Passed: ${data.passed}/${data.total}`);
                // Reload past results to include the new run
                loadPastResults();
              } else if (data.type === 'error') {
                toast.error(`Error: ${data.message}`);
              }
            } catch (e) {
              console.error('Error parsing stream data:', e, line);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error executing tests:', error);
      toast.error(`Execution failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `athena-results-${Date.now()}.json`;
    link.click();
    toast.success('Results exported!');
  };

  const toggleRow = (id) => {
    setExpandedRows(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const handleViewRun = (run) => {
    setResults(run.results);
    setSelectedRun(run.id);
    setExpandedRows({}); // Collapse all rows when switching runs
  };

  const passRate = results.length > 0
    ? ((results.filter(r => r.passed).length / results.length) * 100).toFixed(1)
    : 0;

  const avgScore = results.length > 0
    ? (results.reduce((sum, r) => sum + r.score, 0) / results.length).toFixed(1)
    : 0;

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
        <h2 className="text-3xl font-bold mb-2" data-testid="test-execution-title">Step 5: Test Execution & Results</h2>
        <p className="text-muted-foreground">
          Run tests and evaluate outputs with interactive results
        </p>
      </div>

      {/* Execution Config */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Eval Prompt</CardTitle>
          </CardHeader>
          <CardContent>
            {evalPrompts.length === 0 ? (
              <p className="text-sm text-muted-foreground">No eval prompts available</p>
            ) : (
              <select
                value={selectedEval?.id || ''}
                onChange={(e) => setSelectedEval(evalPrompts.find(ep => ep.id === e.target.value))}
                className="w-full p-2 border border-input rounded-md bg-background text-sm"
                data-testid="eval-select"
              >
                {evalPrompts.map(ep => (
                  <option key={ep.id} value={ep.id}>
                    {ep.dimension} (Score: {ep.quality_score?.toFixed(1)})
                  </option>
                ))}
              </select>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Test Cases</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              {selectedTests.length} / {testCases.length} selected
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Provider Selection */}
            <div className="space-y-2">
              <Label htmlFor="provider-select" className="text-xs">Provider</Label>
              <select
                id="provider-select"
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                className="w-full p-2 border border-input rounded-md bg-background text-sm"
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="google">Google (Gemini)</option>
              </select>
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model-select" className="text-xs">Model</Label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 border border-input rounded-md bg-background text-sm"
              >
                {selectedProvider === 'openai' && (
                  <>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="o3">o3</option>
                    <option value="o3-mini">o3-mini</option>
                  </>
                )}
                {selectedProvider === 'anthropic' && (
                  <>
                    <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
                    <option value="claude-opus-4-5-20251101">Claude Opus 4.5</option>
                  </>
                )}
                {selectedProvider === 'google' && (
                  <>
                    <option value="gemini-2.5-flash-preview-01-21">Gemini 2.5 Flash</option>
                    <option value="gemini-2.5-pro-preview-01-21">Gemini 2.5 Pro</option>
                  </>
                )}
              </select>
            </div>

            {/* Execute Button */}
            <Button
              onClick={handleExecute}
              disabled={loading || !selectedEval || selectedTests.length === 0}
              className="w-full button-hover"
              data-testid="execute-tests-button"
            >
              {loading ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Running...</>
              ) : (
                <><Play className="w-4 h-4 mr-2" />Execute Tests</>
              )}
            </Button>

            {/* Export Button */}
            {results.length > 0 && (
              <Button
                onClick={handleExport}
                variant="outline"
                className="w-full"
                data-testid="export-results-button"
              >
                <Download className="w-4 h-4 mr-2" />
                Export JSON
              </Button>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Execution History */}
      {executionRuns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="w-5 h-5" />
              Execution History
            </CardTitle>
            <CardDescription>
              View results from past test executions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {executionRuns.map((run) => (
                <div
                  key={run.id}
                  onClick={() => handleViewRun(run)}
                  className={`p-4 border rounded-lg cursor-pointer transition-all hover:border-primary ${
                    selectedRun === run.id ? 'border-primary bg-primary/5' : 'border-border'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Clock className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm font-medium">
                          {run.timestamp.toLocaleString()}
                        </span>
                        {selectedRun === run.id && (
                          <Badge variant="default" className="text-xs">Current</Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Eval: {evalPrompts.find(ep => ep.id === run.evalPromptId)?.dimension || 'Unknown'}
                      </p>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-center">
                        <p className="font-medium text-primary">{run.passed}/{run.totalTests}</p>
                        <p className="text-xs text-muted-foreground">Passed</p>
                      </div>
                      <div className="text-center">
                        <p className="font-medium text-primary">{run.avgScore}/5</p>
                        <p className="text-xs text-muted-foreground">Avg Score</p>
                      </div>
                      <div className="text-center">
                        <p className="font-medium text-primary">{((run.passed / run.totalTests) * 100).toFixed(0)}%</p>
                        <p className="text-xs text-muted-foreground">Pass Rate</p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary Stats */}
      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-primary">{passRate}%</p>
                <p className="text-sm text-muted-foreground mt-1">Pass Rate</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-primary">{avgScore}/5</p>
                <p className="text-sm text-muted-foreground mt-1">Average Score</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-primary">{results.length}</p>
                <p className="text-sm text-muted-foreground mt-1">Tests Run</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Results Table */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Test Results</CardTitle>
                <CardDescription>Click rows to expand details</CardDescription>
              </div>
              {selectedRun && executionRuns.length > 0 && (
                <Badge variant="outline" className="text-xs">
                  {executionRuns.find(r => r.id === selectedRun)?.timestamp.toLocaleString() || 'Current Run'}
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12"></TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Input</TableHead>
                    <TableHead>Output Preview</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.map((result) => (
                    <>
                      <TableRow
                        key={result.id}
                        className="cursor-pointer hover:bg-muted/50"
                        onClick={() => toggleRow(result.id)}
                        data-testid={`result-row-${result.id}`}
                      >
                        <TableCell>
                          <ChevronDown
                            className={`w-4 h-4 transition-transform ${
                              expandedRows[result.id] ? 'rotate-180' : ''
                            }`}
                          />
                        </TableCell>
                        <TableCell>
                          {result.passed ? (
                            <CheckCircle2 className="w-5 h-5 text-green-600" />
                          ) : (
                            <XCircle className="w-5 h-5 text-red-600" />
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant={result.score >= 3 ? 'default' : 'destructive'}>
                            {result.score}/5
                          </Badge>
                        </TableCell>
                        <TableCell className="max-w-xs truncate font-mono text-xs">
                          {result.input_text}
                        </TableCell>
                        <TableCell className="max-w-xs truncate text-xs">
                          {result.output.substring(0, 50)}...
                        </TableCell>
                      </TableRow>
                      {expandedRows[result.id] && (
                        <TableRow>
                          <TableCell colSpan={5} className="bg-muted/30">
                            <div className="p-4 space-y-4">
                              <div>
                                <p className="text-xs font-medium text-muted-foreground mb-1">Full Input:</p>
                                <p className="text-sm font-mono bg-background p-3 rounded border">
                                  {result.input_text}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs font-medium text-muted-foreground mb-1">Full Output:</p>
                                <p className="text-sm bg-background p-3 rounded border whitespace-pre-wrap">
                                  {result.output}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs font-medium text-muted-foreground mb-1">Evaluation Result:</p>
                                <pre className="text-xs bg-background p-3 rounded border overflow-x-auto">
                                  {JSON.stringify(result.eval_result, null, 2)}
                                </pre>
                              </div>
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                    </>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {results.length === 0 && executionRuns.length === 0 && (
        <Card>
          <CardContent className="py-12">
            <div className="text-center text-muted-foreground">
              <Play className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No results yet. Execute tests to see results here.</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}