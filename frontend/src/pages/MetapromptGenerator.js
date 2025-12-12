import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Badge } from '../components/ui/badge';
import { Wand2, Loader2, Copy, Check } from 'lucide-react';
import axios from 'axios';
import { API } from '../App';

const MetapromptGenerator = () => {
  const [promptText, setPromptText] = useState('');
  const [desiredBehavior, setDesiredBehavior] = useState('');
  const [undesiredBehavior, setUndesiredBehavior] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    if (!promptText.trim() || !desiredBehavior.trim() || !undesiredBehavior.trim()) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API}/generate-metaprompt`, {
        prompt_text: promptText,
        desired_behavior: desiredBehavior,
        undesired_behavior: undesiredBehavior
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate metaprompt');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getEditTypeColor = (type) => {
    switch (type) {
      case 'add':
        return 'bg-green-500';
      case 'delete':
        return 'bg-red-500';
      case 'modify':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Wand2 className="w-8 h-8" />
          Metaprompt Generator
        </h1>
        <p className="text-muted-foreground mt-2">
          Use GPT-5's metaprompting capability to generate minimal edits that improve your prompt's behavior.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Section */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Current Prompt</CardTitle>
              <CardDescription>
                Enter the prompt that needs improvement
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Your current prompt..."
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                className="min-h-[200px] font-mono text-sm"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Behavior Description</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="desired">Desired Behavior</Label>
                <Input
                  id="desired"
                  placeholder="What should the prompt do?"
                  value={desiredBehavior}
                  onChange={(e) => setDesiredBehavior(e.target.value)}
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="undesired">Current Undesired Behavior</Label>
                <Input
                  id="undesired"
                  placeholder="What is it doing instead?"
                  value={undesiredBehavior}
                  onChange={(e) => setUndesiredBehavior(e.target.value)}
                  className="mt-1"
                />
              </div>
            </CardContent>
          </Card>

          <Button 
            onClick={handleGenerate} 
            disabled={loading || !promptText.trim() || !desiredBehavior.trim() || !undesiredBehavior.trim()}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Wand2 className="mr-2 h-4 w-4" />
                Generate Improvements
              </>
            )}
          </Button>
        </div>

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
              {/* Analysis */}
              {result.analysis && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{result.analysis}</p>
                  </CardContent>
                </Card>
              )}

              {/* Suggested Edits */}
              {result.suggested_edits && result.suggested_edits.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Suggested Edits</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {result.suggested_edits.map((edit, index) => (
                      <div key={index} className="p-3 rounded-lg border bg-card space-y-2">
                        <div className="flex items-center gap-2">
                          <Badge className={`${getEditTypeColor(edit.type)} text-white`}>
                            {edit.type?.toUpperCase()}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {edit.location}
                          </span>
                        </div>
                        <div className="p-2 bg-muted rounded font-mono text-xs">
                          {edit.content}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          <strong>Why:</strong> {edit.rationale}
                        </p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Improved Prompt */}
              {result.improved_prompt && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center justify-between">
                      <span>Improved Prompt</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCopy(result.improved_prompt)}
                      >
                        {copied ? (
                          <>
                            <Check className="w-4 h-4 mr-1" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4 mr-1" />
                            Copy
                          </>
                        )}
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-xs bg-muted p-3 rounded overflow-x-auto whitespace-pre-wrap font-mono">
                      {result.improved_prompt}
                    </pre>
                  </CardContent>
                </Card>
              )}

              {/* Expected Improvement */}
              {result.expected_improvement && (
                <Card className="border-green-200 dark:border-green-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <span>✨</span>
                      Expected Improvement
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{result.expected_improvement}</p>
                  </CardContent>
                </Card>
              )}

              {/* Cost */}
              {result.cost && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Generation Cost</CardTitle>
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
                <Wand2 className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p>Fill in all fields and click "Generate Improvements" to see suggestions</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Info Section */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>How Metaprompting Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <h3 className="font-semibold mb-2">What is Metaprompting?</h3>
              <p className="text-muted-foreground">
                Metaprompting uses GPT-5's self-awareness to suggest minimal, targeted edits to your prompt.
                Instead of rewriting the entire prompt, it identifies specific additions, deletions, or modifications
                that will improve behavior while keeping your original structure intact.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Best Use Cases:</h3>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Fixing specific behavioral issues</li>
                <li>• Improving instruction following</li>
                <li>• Enhancing output quality</li>
                <li>• Iterative prompt refinement</li>
                <li>• Debugging prompt problems</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default MetapromptGenerator;
