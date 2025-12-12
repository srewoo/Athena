import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { GitCompare, Loader2, Trophy, TrendingUp, TrendingDown, Minus, ArrowLeft } from 'lucide-react';
import axios from 'axios';
import { useToast } from '../hooks/use-toast';
import { API } from '../App';

const ABTesting = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [promptA, setPromptA] = useState('');
  const [promptB, setPromptB] = useState('');
  const [testName, setTestName] = useState('');
  const [description, setDescription] = useState('');
  const [evaluationMode, setEvaluationMode] = useState('standard');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleRunTest = async () => {
    if (!promptA.trim() || !promptB.trim()) {
      toast({
        variant: "destructive",
        title: "Missing Prompts",
        description: "Please enter both Prompt A and Prompt B"
      });
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API}/ab-test`, {
        prompt_a: promptA,
        prompt_b: promptB,
        evaluation_mode: evaluationMode,
        test_name: testName || undefined,
        description: description || undefined
      }, {
        timeout: 300000 // 5 minute timeout for two evaluations
      });

      setResult(response.data);
      toast({
        title: "A/B Test Complete!",
        description: `Winner: Prompt ${response.data.comparison.winner} (${response.data.comparison.confidence} confidence)`
      });
    } catch (error) {
      const errorMessage = error.code === 'ECONNABORTED'
        ? "Request timed out. Try with shorter prompts or a faster evaluation mode."
        : error.response?.data?.detail || "An error occurred during A/B testing";
      
      toast({
        variant: "destructive",
        title: "A/B Test Failed",
        description: errorMessage
      });
    } finally {
      setLoading(false);
    }
  };

  const getWinnerColor = (winner) => {
    if (winner === 'Tie') return 'text-slate-400';
    return winner === 'A' ? 'text-blue-400' : 'text-purple-400';
  };

  const getConfidenceBadge = (confidence) => {
    const colors = {
      'High': 'bg-green-600',
      'Medium': 'bg-yellow-600',
      'Low': 'bg-slate-600'
    };
    return colors[confidence] || 'bg-slate-600';
  };

  return (
    <div className="min-h-screen p-6 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              onClick={() => navigate("/")}
              className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div>
              <h1 className="text-4xl font-bold text-white flex items-center gap-3">
                <GitCompare className="w-10 h-10 text-blue-400" />
                A/B Testing
              </h1>
              <p className="text-slate-400 mt-2">
                Compare two prompts side-by-side with statistical analysis
              </p>
            </div>
          </div>
        </div>

        {/* Input Section */}
        <div className="grid gap-6 md:grid-cols-2 mb-6">
          {/* Prompt A */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Badge className="bg-blue-600">A</Badge>
                Prompt A (Control)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter your control prompt..."
                value={promptA}
                onChange={(e) => setPromptA(e.target.value)}
                className="min-h-[300px] font-mono text-sm bg-slate-900 border-slate-700 text-white"
              />
            </CardContent>
          </Card>

          {/* Prompt B */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Badge className="bg-purple-600">B</Badge>
                Prompt B (Variant)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter your variant prompt..."
                value={promptB}
                onChange={(e) => setPromptB(e.target.value)}
                className="min-h-[300px] font-mono text-sm bg-slate-900 border-slate-700 text-white"
              />
            </CardContent>
          </Card>
        </div>

        {/* Test Configuration */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">Test Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Test Name (Optional)</label>
                <Input
                  placeholder="e.g., Customer Service Prompt Optimization"
                  value={testName}
                  onChange={(e) => setTestName(e.target.value)}
                  className="bg-slate-900 border-slate-700 text-white"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Evaluation Mode</label>
                <select
                  value={evaluationMode}
                  onChange={(e) => setEvaluationMode(e.target.value)}
                  className="w-full p-2 rounded-md bg-slate-900 border border-slate-700 text-white"
                >
                  <option value="quick">Quick (10 criteria, ~60s)</option>
                  <option value="standard">Standard (50 criteria, ~120s)</option>
                  <option value="deep">Deep Analysis (~180s)</option>
                  <option value="agentic">Agentic Workflow</option>
                  <option value="long_context">Long Context</option>
                </select>
              </div>
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Description (Optional)</label>
              <Input
                placeholder="Brief description of what you're testing..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="bg-slate-900 border-slate-700 text-white"
              />
            </div>
            <Button
              onClick={handleRunTest}
              disabled={loading || !promptA.trim() || !promptB.trim()}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              size="lg"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Running A/B Test...
                </>
              ) : (
                <>
                  <GitCompare className="mr-2 h-5 w-5" />
                  Run A/B Test
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Winner Card */}
            <Card className="bg-gradient-to-r from-slate-800 to-slate-900 border-slate-700">
              <CardContent className="pt-6">
                <div className="text-center">
                  <Trophy className={`w-16 h-16 mx-auto mb-4 ${getWinnerColor(result.comparison.winner)}`} />
                  <h2 className="text-3xl font-bold text-white mb-2">
                    {result.comparison.winner === 'Tie' ? 'No Clear Winner' : `Prompt ${result.comparison.winner} Wins!`}
                  </h2>
                  <Badge className={`${getConfidenceBadge(result.comparison.confidence)} text-white text-lg px-4 py-1`}>
                    {result.comparison.confidence} Confidence
                  </Badge>
                  <p className="text-slate-300 mt-4 max-w-2xl mx-auto">
                    {result.comparison.recommendation}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Score Comparison */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Prompt A Score */}
              <Card className="bg-slate-800/50 border-blue-500/30">
                <CardHeader>
                  <CardTitle className="text-white flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Badge className="bg-blue-600">A</Badge>
                      Prompt A
                    </span>
                    <span className="text-3xl">{result.prompt_a.total_score}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Progress value={result.prompt_a.percentage} className="h-3 mb-2" />
                  <p className="text-slate-400 text-sm">{result.prompt_a.percentage}% Quality</p>
                </CardContent>
              </Card>

              {/* Prompt B Score */}
              <Card className="bg-slate-800/50 border-purple-500/30">
                <CardHeader>
                  <CardTitle className="text-white flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Badge className="bg-purple-600">B</Badge>
                      Prompt B
                    </span>
                    <span className="text-3xl">{result.prompt_b.total_score}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Progress value={result.prompt_b.percentage} className="h-3 mb-2" />
                  <p className="text-slate-400 text-sm">{result.prompt_b.percentage}% Quality</p>
                </CardContent>
              </Card>
            </div>

            {/* Score Difference */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Score Difference</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-center gap-4">
                  {result.comparison.score_difference > 0 ? (
                    <TrendingUp className="w-8 h-8 text-green-400" />
                  ) : result.comparison.score_difference < 0 ? (
                    <TrendingDown className="w-8 h-8 text-red-400" />
                  ) : (
                    <Minus className="w-8 h-8 text-slate-400" />
                  )}
                  <div className="text-center">
                    <p className="text-4xl font-bold text-white">
                      {result.comparison.score_difference > 0 ? '+' : ''}{result.comparison.score_difference}
                    </p>
                    <p className="text-slate-400">
                      ({result.comparison.score_difference_percent > 0 ? '+' : ''}{result.comparison.score_difference_percent}%)
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Category Comparison */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Category Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {result.comparison.category_comparison.map((cat, idx) => (
                    <div key={idx} className="bg-slate-900 p-4 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white font-medium">{cat.category}</span>
                        <Badge className={cat.winner === 'A' ? 'bg-blue-600' : cat.winner === 'B' ? 'bg-purple-600' : 'bg-slate-600'}>
                          {cat.winner}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <p className="text-slate-400">Prompt A</p>
                          <p className="text-white font-bold">{cat.prompt_a_score}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Prompt B</p>
                          <p className="text-white font-bold">{cat.prompt_b_score}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Difference</p>
                          <p className={`font-bold ${cat.difference > 0 ? 'text-green-400' : cat.difference < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                            {cat.difference > 0 ? '+' : ''}{cat.difference}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Top Improvements */}
            {result.comparison.top_improvements_b_over_a.length > 0 && (
              <Card className="bg-slate-800/50 border-green-500/30">
                <CardHeader>
                  <CardTitle className="text-white">Where Prompt B Excels</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.comparison.top_improvements_b_over_a.map((imp, idx) => (
                      <li key={idx} className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">{imp.criterion}</span>
                        <Badge className="bg-green-600">+{imp.improvement}</Badge>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {result.comparison.top_improvements_a_over_b.length > 0 && (
              <Card className="bg-slate-800/50 border-blue-500/30">
                <CardHeader>
                  <CardTitle className="text-white">Where Prompt A Excels</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.comparison.top_improvements_a_over_b.map((imp, idx) => (
                      <li key={idx} className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">{imp.criterion}</span>
                        <Badge className="bg-blue-600">+{imp.improvement}</Badge>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {/* Cost */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Test Cost</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-slate-400 text-sm">Prompt A</p>
                    <p className="text-white font-bold">${result.cost.prompt_a_cost.toFixed(6)}</p>
                  </div>
                  <div>
                    <p className="text-slate-400 text-sm">Prompt B</p>
                    <p className="text-white font-bold">${result.cost.prompt_b_cost.toFixed(6)}</p>
                  </div>
                  <div>
                    <p className="text-slate-400 text-sm">Total</p>
                    <p className="text-white font-bold">${result.cost.total_cost.toFixed(6)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default ABTesting;
