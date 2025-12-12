import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Badge } from '../components/ui/badge';
import { FileCode, Loader2, Lightbulb } from 'lucide-react';
import axios from 'axios';
import { API } from '../App';

const DelimiterAnalyzer = () => {
  const [promptText, setPromptText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!promptText.trim()) {
      setError('Please enter a prompt to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API}/analyze-delimiters`, {
        prompt_text: promptText
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze delimiters');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 4) return 'bg-green-500';
    if (score >= 3) return 'bg-blue-500';
    if (score >= 2) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <FileCode className="w-8 h-8" />
          Delimiter Strategy Analyzer
        </h1>
        <p className="text-muted-foreground mt-2">
          Analyze and optimize your prompt's delimiter strategy. Based on GPT-4.1 and Claude best practices.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle>Enter Your Prompt</CardTitle>
            <CardDescription>
              Paste the prompt you want to analyze for delimiter usage
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter your prompt here..."
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              className="min-h-[400px] font-mono text-sm"
            />
            <Button 
              onClick={handleAnalyze} 
              disabled={loading || !promptText.trim()}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <FileCode className="mr-2 h-4 w-4" />
                  Analyze Delimiters
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        <div className="space-y-4">
          {error && (
            <Card className="border-red-200 dark:border-red-800">
              <CardContent className="pt-6">
                <p className="text-red-600 dark:text-red-400">{error}</p>
              </CardContent>
            </Card>
          )}

          {result && (
            <>
              {/* Score Card */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Delimiter Quality</span>
                    <Badge className={`${getScoreColor(result.quality_score)} text-white`}>
                      {result.quality_score}/5
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-sm font-medium mb-1">Current Strategy</p>
                    <Badge variant="outline" className="text-sm">
                      {result.current_strategy?.toUpperCase()}
                    </Badge>
                  </div>
                  
                  {result.optimal_format && (
                    <div>
                      <p className="text-sm font-medium mb-1">Recommended Format</p>
                      <Badge variant="default" className="text-sm">
                        {result.optimal_format.toUpperCase()}
                      </Badge>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Strengths */}
              {Array.isArray(result.strengths) && result.strengths.length > 0 && (
                <Card className="border-green-200 dark:border-green-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <span>✅</span>
                      Strengths
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1">
                      {result.strengths.map((strength, index) => (
                        <li key={index} className="text-sm text-muted-foreground pl-5 relative">
                          <span className="absolute left-0">•</span>
                          {typeof strength === 'string' ? strength : JSON.stringify(strength)}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {/* Weaknesses */}
              {Array.isArray(result.weaknesses) && result.weaknesses.length > 0 && (
                <Card className="border-yellow-200 dark:border-yellow-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <span>⚠️</span>
                      Areas for Improvement
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1">
                      {result.weaknesses.map((weakness, index) => (
                        <li key={index} className="text-sm text-muted-foreground pl-5 relative">
                          <span className="absolute left-0">•</span>
                          {typeof weakness === 'string' ? weakness : JSON.stringify(weakness)}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {/* Recommendations */}
              {Array.isArray(result.recommendations) && result.recommendations.length > 0 && (
                <Card className="border-blue-200 dark:border-blue-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Lightbulb className="w-4 h-4" />
                      Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {result.recommendations.map((rec, index) => (
                        <li key={index} className="text-sm text-muted-foreground pl-5 relative">
                          <span className="absolute left-0">•</span>
                          {typeof rec === 'string' ? rec : JSON.stringify(rec)}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {/* Example Improvement */}
              {result.example_improvement && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Example Improvement</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {typeof result.example_improvement === 'string' ? (
                      <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                        {result.example_improvement}
                      </pre>
                    ) : result.example_improvement.before && result.example_improvement.after ? (
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-semibold mb-1 text-red-600">Before:</p>
                          <pre className="text-xs bg-muted p-3 rounded overflow-x-auto border-l-2 border-red-500">
                            {result.example_improvement.before}
                          </pre>
                        </div>
                        <div>
                          <p className="text-xs font-semibold mb-1 text-green-600">After:</p>
                          <pre className="text-xs bg-muted p-3 rounded overflow-x-auto border-l-2 border-green-500">
                            {result.example_improvement.after}
                          </pre>
                        </div>
                      </div>
                    ) : (
                      <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                        {JSON.stringify(result.example_improvement, null, 2)}
                      </pre>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Cost */}
              {result.cost && typeof result.cost.total_cost === 'number' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Analysis Cost</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="font-bold">${result.cost.total_cost.toFixed(6)} USD</p>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {!result && !error && !loading && (
            <Card className="border-dashed">
              <CardContent className="pt-6 text-center text-muted-foreground">
                <FileCode className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p>Enter a prompt and click "Analyze Delimiters" to see results</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Info Section */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Delimiter Best Practices</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <h3 className="font-semibold mb-2">✅ XML (Best for Claude)</h3>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Precise content wrapping</li>
                <li>• Metadata in tags</li>
                <li>• Excellent for nesting</li>
                <li>• Clear start/end markers</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">✅ Markdown (Best for OpenAI)</h3>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Clear hierarchy</li>
                <li>• Headers and sections</li>
                <li>• Code blocks</li>
                <li>• Lists and formatting</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">❌ JSON (Avoid for long context)</h3>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Poor long context performance</li>
                <li>• Verbose syntax</li>
                <li>• Character escaping needed</li>
                <li>• Use pipe format instead</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DelimiterAnalyzer;
