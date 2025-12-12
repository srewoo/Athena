import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { AlertTriangle, Loader2 } from 'lucide-react';
import ContradictionAlert from '../components/ContradictionAlert';
import axios from 'axios';
import { API } from '../App';

const ContradictionDetector = () => {
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
      const response = await axios.post(`${API}/detect-contradictions`, {
        prompt_text: promptText
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze contradictions');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <AlertTriangle className="w-8 h-8" />
          Contradiction Detector
        </h1>
        <p className="text-muted-foreground mt-2">
          Detect conflicting or contradictory instructions in your prompts. Based on GPT-5 best practices.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle>Enter Your Prompt</CardTitle>
            <CardDescription>
              Paste the prompt you want to check for contradictions
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
                  <AlertTriangle className="mr-2 h-4 w-4" />
                  Detect Contradictions
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
              <ContradictionAlert contradictionAnalysis={result} />
              
              {result.cost && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Analysis Cost</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground">Input Tokens</p>
                        <p className="font-semibold">{result.cost.input_tokens}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Output Tokens</p>
                        <p className="font-semibold">{result.cost.output_tokens}</p>
                      </div>
                      <div className="col-span-2 pt-2 border-t">
                        <p className="text-muted-foreground">Total Cost</p>
                        <p className="font-bold text-lg">
                          ${result.cost.total_cost.toFixed(6)} USD
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {!result && !error && !loading && (
            <Card className="border-dashed">
              <CardContent className="pt-6 text-center text-muted-foreground">
                <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p>Enter a prompt and click "Detect Contradictions" to see results</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Info Section */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>What This Tool Detects</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold mb-2">Types of Contradictions:</h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Direct contradictions ("Never X" vs "Always X")</li>
                <li>• Conflicting priorities (speed vs accuracy)</li>
                <li>• Ambiguous conditional logic</li>
                <li>• Inconsistent formatting requirements</li>
                <li>• Contradictory tone or style instructions</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Why It Matters:</h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• GPT-5 is highly sensitive to instruction consistency</li>
                <li>• Contradictions waste reasoning tokens</li>
                <li>• Models may pick instructions randomly</li>
                <li>• Reduces output quality and reliability</li>
                <li>• Can cause unpredictable behavior</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ContradictionDetector;
