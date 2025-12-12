import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../components/ui/accordion";
import { ArrowLeft, CheckCircle, Sparkles, Play, DollarSign, History, GitCompare, Settings, Download, AlertCircle, Lightbulb, AlertTriangle, FileCode, Wand2, Zap, Target, Brain, Layers } from "lucide-react";
import { Badge } from "../components/ui/badge";
import { BASE_URL } from "../App";

export default function Help() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-5xl mx-auto">
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
              <h1 className="text-4xl font-bold text-white">Help & Documentation</h1>
              <p className="text-slate-400 text-lg">Learn how to use PromptCritic effectively</p>
            </div>
          </div>
        </div>

        {/* Quick Start Guide */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              Quick Start Guide
            </CardTitle>
            <CardDescription className="text-slate-400">
              Get started with PromptCritic in 3 easy steps
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex gap-4 items-start">
                <div className="bg-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 font-bold">
                  1
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">Configure Your LLM Settings</h3>
                  <p className="text-slate-300 text-sm">
                    Click the <Settings className="inline h-3 w-3" /> <strong>Settings</strong> button and enter your API key for OpenAI, Claude, or Gemini.
                    Choose your preferred provider and optionally specify a model name.
                  </p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="bg-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 font-bold">
                  2
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">Enter Your Prompt</h3>
                  <p className="text-slate-300 text-sm">
                    Type or paste your prompt in the text area. Be as detailed or concise as you need - the AI will evaluate it across 50 expert criteria.
                  </p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="bg-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 font-bold">
                  3
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">Evaluate & Improve</h3>
                  <p className="text-slate-300 text-sm">
                    Click <CheckCircle className="inline h-3 w-3" /> <strong>Evaluate Prompt</strong> to get your score, suggestions, and an AI-powered rewrite to improve your prompt instantly.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Features Overview */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">Features Overview</CardTitle>
            <CardDescription className="text-slate-400">
              Explore what PromptCritic can do for you
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Feature 1 */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-blue-400" />
                  <h3 className="text-white font-semibold">50-Criteria Evaluation</h3>
                </div>
                <p className="text-slate-300 text-sm">
                  Comprehensive analysis across 7 categories including Core Fundamentals, Modern Best Practices, Provider Optimization, Advanced Techniques, Agentic Patterns, Context Management, and Safety & Reliability.
                </p>
              </div>

              {/* Feature 2 */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="h-5 w-5 text-purple-400" />
                  <h3 className="text-white font-semibold">AI-Powered Rewriting</h3>
                </div>
                <p className="text-slate-300 text-sm">
                  Get an improved version of your prompt instantly, based on evaluation feedback and best practices.
                </p>
              </div>

              {/* Feature 3 */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Play className="h-5 w-5 text-purple-400" />
                  <h3 className="text-white font-semibold">Prompt Playground</h3>
                </div>
                <p className="text-slate-300 text-sm">
                  Test your prompts with sample input before evaluation. See live results and cost estimates.
                </p>
              </div>

              {/* Feature 4 */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="h-5 w-5 text-green-400" />
                  <h3 className="text-white font-semibold">Cost Calculator</h3>
                </div>
                <p className="text-slate-300 text-sm">
                  Track API costs with detailed token usage and pricing breakdown for every operation.
                </p>
              </div>

              {/* Feature 5 */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <History className="h-5 w-5 text-blue-400" />
                  <h3 className="text-white font-semibold">Evaluation History</h3>
                </div>
                <p className="text-slate-300 text-sm">
                  Access all your past evaluations, track improvements, and revisit previous analyses.
                </p>
              </div>

              {/* Feature 6 */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <GitCompare className="h-5 w-5 text-blue-400" />
                  <h3 className="text-white font-semibold">Compare Evaluations</h3>
                </div>
                <p className="text-slate-300 text-sm">
                  Compare multiple prompts side-by-side to understand what works best.
                </p>
              </div>

              {/* Feature 7 - NEW */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="h-5 w-5 text-orange-400" />
                  <h3 className="text-white font-semibold">Contradiction Detector</h3>
                  <Badge className="bg-green-600 text-xs">New</Badge>
                </div>
                <p className="text-slate-300 text-sm">
                  Detect conflicting instructions in your prompts. Critical for GPT-5 which is highly sensitive to contradictions.
                </p>
              </div>

              {/* Feature 8 - NEW */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <FileCode className="h-5 w-5 text-cyan-400" />
                  <h3 className="text-white font-semibold">Delimiter Analyzer</h3>
                  <Badge className="bg-green-600 text-xs">New</Badge>
                </div>
                <p className="text-slate-300 text-sm">
                  Analyze and optimize your delimiter strategy (XML, Markdown, JSON) based on GPT-4.1 research.
                </p>
              </div>

              {/* Feature 9 - NEW */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Wand2 className="h-5 w-5 text-pink-400" />
                  <h3 className="text-white font-semibold">Metaprompt Generator</h3>
                  <Badge className="bg-green-600 text-xs">New</Badge>
                </div>
                <p className="text-slate-300 text-sm">
                  GPT-5 metaprompting: Get minimal edit suggestions to improve prompt behavior based on desired outcomes.
                </p>
              </div>

              {/* Feature 10 - NEW */}
              <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-5 w-5 text-blue-400" />
                  <h3 className="text-white font-semibold">5 Evaluation Modes</h3>
                  <Badge className="bg-green-600 text-xs">New</Badge>
                </div>
                <p className="text-slate-300 text-sm">
                  Choose from Quick (10 criteria), Standard (50), Deep, Agentic, or Long Context modes for tailored analysis.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Detailed Guides */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">How To Use Each Feature</CardTitle>
            <CardDescription className="text-slate-400">
              Step-by-step guides for all features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible className="space-y-2">
              {/* Guide 1 */}
              <AccordionItem value="item-1" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium">How to Evaluate a Prompt</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <ol className="list-decimal list-inside space-y-2">
                    <li>Navigate to the Dashboard (home page)</li>
                    <li>Enter or paste your prompt in the text area</li>
                    <li>Click the <strong>"Evaluate Prompt"</strong> button</li>
                    <li>Wait for the AI analysis (usually takes 5-15 seconds)</li>
                    <li>Review your score out of 250 points</li>
                    <li>Read the top refinement suggestions</li>
                    <li>Click <strong>"View Full Details"</strong> to see all 50 criteria scores</li>
                  </ol>
                  <div className="bg-blue-900/20 border border-blue-700 p-3 rounded-lg mt-3">
                    <p className="text-blue-300 text-sm">
                      <strong>Tip:</strong> Scores above 140 are excellent, 100-140 are good, 70-100 are fair, and below 70 need improvement.
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 2 */}
              <AccordionItem value="item-2" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium">How to Use AI Rewrite</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <ol className="list-decimal list-inside space-y-2">
                    <li>First, evaluate a prompt (see above)</li>
                    <li>After evaluation completes, click the <strong>"AI Rewrite"</strong> button</li>
                    <li>The AI will analyze low-scoring areas and suggestions</li>
                    <li>Review the improved prompt in the modal</li>
                    <li>See key improvements and rationale</li>
                    <li>Click <strong>"Copy"</strong> to copy the improved prompt</li>
                    <li>Or click <strong>"Use This Prompt"</strong> to replace your original and re-evaluate</li>
                  </ol>
                  <div className="bg-purple-900/20 border border-purple-700 p-3 rounded-lg mt-3">
                    <p className="text-purple-300 text-sm">
                      <strong>Tip:</strong> Use the rewrite feature iteratively - evaluate, rewrite, evaluate again to continuously improve your prompts!
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 3 */}
              <AccordionItem value="item-3" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium">How to Use the Playground</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <ol className="list-decimal list-inside space-y-2">
                    <li>Click <strong>"Playground"</strong> button from the dashboard</li>
                    <li>Enter your prompt template (use <code className="bg-slate-800 px-1 rounded">{`{input}`}</code> as a placeholder)</li>
                    <li>Provide test data in the "Test Input" field</li>
                    <li>Click <strong>"Test Prompt"</strong></li>
                    <li>View the AI's response and cost breakdown</li>
                    <li>Iterate and refine your prompt based on results</li>
                  </ol>
                  <div className="bg-purple-900/20 border border-purple-700 p-3 rounded-lg mt-3">
                    <p className="text-purple-300 text-sm">
                      <strong>Example:</strong> Template: "Write a {`{input}`}" | Test Input: "short story about a robot" | Result: Full story generated
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 4 */}
              <AccordionItem value="item-4" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium">Understanding the Cost Calculator</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <p>Every evaluation, rewrite, and playground test shows cost information:</p>
                  <ul className="list-disc list-inside space-y-2 ml-4">
                    <li><strong>Input Tokens:</strong> Number of tokens in your prompt/request</li>
                    <li><strong>Output Tokens:</strong> Number of tokens in the AI's response</li>
                    <li><strong>Input Cost:</strong> Cost for processing your prompt</li>
                    <li><strong>Output Cost:</strong> Cost for generating the response</li>
                    <li><strong>Total Cost:</strong> Combined cost in USD</li>
                  </ul>
                  <div className="bg-green-900/20 border border-green-700 p-3 rounded-lg mt-3">
                    <p className="text-green-300 text-sm">
                      <strong>Cost Optimization:</strong> Shorter prompts cost less. Use the playground to test before full evaluation. Consider using cheaper models for testing.
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 5 */}
              <AccordionItem value="item-5" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium">Viewing and Managing History</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <ol className="list-decimal list-inside space-y-2">
                    <li>Click <strong>"History"</strong> button from the dashboard</li>
                    <li>Browse all your past evaluations</li>
                    <li>Click any evaluation to view full details</li>
                    <li>Export evaluations as PDF or JSON</li>
                    <li>Delete evaluations you no longer need</li>
                    <li>Compare multiple evaluations side-by-side</li>
                  </ol>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 6 */}
              <AccordionItem value="item-6" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium">Configuring LLM Settings</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <ol className="list-decimal list-inside space-y-2">
                    <li>Click <Settings className="inline h-3 w-3" /> <strong>"Settings"</strong> button</li>
                    <li>Select your LLM provider (OpenAI, Claude, or Gemini)</li>
                    <li>Enter your API key</li>
                    <li>Optionally specify a custom model name</li>
                    <li>Click <strong>"Save Configuration"</strong></li>
                  </ol>
                  <div className="bg-amber-900/20 border border-amber-700 p-3 rounded-lg mt-3">
                    <p className="text-amber-300 text-sm">
                      <strong>Security:</strong> Your API key is only stored in your browser and sent directly to the LLM provider. It's never stored on our servers.
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 7 - NEW */}
              <AccordionItem value="item-7" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium flex items-center gap-2">
                    <Zap className="h-4 w-4 text-yellow-500" />
                    Choosing an Evaluation Mode
                    <Badge className="bg-green-600 text-xs">New</Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-3 pt-2">
                  <p>Select the right evaluation depth for your needs:</p>
                  <div className="space-y-3">
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Zap className="h-4 w-4 text-yellow-500" />
                        <strong className="text-white">Quick Scan (10 criteria, ~30s)</strong>
                      </div>
                      <p className="text-sm">Fast feedback for rapid iteration. Best for: initial drafts, quick checks, testing variations.</p>
                    </div>
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Target className="h-4 w-4 text-blue-500" />
                        <strong className="text-white">Standard (50 criteria, ~60s) - Default</strong>
                      </div>
                      <p className="text-sm">Comprehensive analysis across all categories. Best for: general prompts, balanced evaluation, most use cases.</p>
                    </div>
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Brain className="h-4 w-4 text-purple-500" />
                        <strong className="text-white">Deep Analysis (50+ criteria, ~90s)</strong>
                      </div>
                      <p className="text-sm">Includes semantic coherence, edge cases, scalability analysis. Best for: production prompts, critical applications.</p>
                    </div>
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Layers className="h-4 w-4 text-green-500" />
                        <strong className="text-white">Agentic Workflow (50 criteria, ~60s)</strong>
                      </div>
                      <p className="text-sm">Focus on tool use, planning, multi-step reasoning. Best for: agent systems, function calling, complex workflows.</p>
                    </div>
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <FileCode className="h-4 w-4 text-indigo-500" />
                        <strong className="text-white">Long Context (50 criteria, ~60s)</strong>
                      </div>
                      <p className="text-sm">Optimized for 50K+ tokens, document processing, RAG systems. Best for: large documents, retrieval applications.</p>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 8 - NEW */}
              <AccordionItem value="item-8" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    Using the Contradiction Detector
                    <Badge className="bg-green-600 text-xs">New</Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <p>Detect conflicting instructions that confuse modern LLMs like GPT-5:</p>
                  <ol className="list-decimal list-inside space-y-2 mt-2">
                    <li>Click the <strong className="text-orange-400">"Contradictions"</strong> button from the dashboard</li>
                    <li>Paste your prompt in the text area</li>
                    <li>Click <strong>"Detect Contradictions"</strong></li>
                    <li>Review detected contradictions with severity ratings (High/Medium/Low)</li>
                    <li>See which instructions conflict with each other</li>
                    <li>Apply suggested fixes to resolve contradictions</li>
                  </ol>
                  <div className="bg-orange-900/20 border border-orange-700 p-3 rounded-lg mt-3">
                    <p className="text-orange-300 text-sm">
                      <strong>Why it matters:</strong> GPT-5 is highly sensitive to contradictory instructions. Even subtle conflicts can cause unpredictable behavior.
                    </p>
                  </div>
                  <div className="bg-slate-800 p-3 rounded-lg mt-2">
                    <p className="text-sm mb-1"><strong className="text-white">Example Contradiction:</strong></p>
                    <p className="text-xs text-red-400">"Be concise" vs "Provide detailed explanations"</p>
                    <p className="text-xs text-green-400 mt-1">✓ Fix: "Be concise but thorough in key areas"</p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 9 - NEW */}
              <AccordionItem value="item-9" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium flex items-center gap-2">
                    <FileCode className="h-4 w-4 text-cyan-500" />
                    Using the Delimiter Analyzer
                    <Badge className="bg-green-600 text-xs">New</Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <p>Optimize your delimiter strategy based on GPT-4.1 research:</p>
                  <ol className="list-decimal list-inside space-y-2 mt-2">
                    <li>Click the <strong className="text-cyan-400">"Delimiters"</strong> button from the dashboard</li>
                    <li>Paste your prompt</li>
                    <li>Click <strong>"Analyze Delimiters"</strong></li>
                    <li>See your current strategy (XML, Markdown, JSON, or Mixed)</li>
                    <li>Review quality score (1-5 stars)</li>
                    <li>Read strengths and weaknesses</li>
                    <li>View recommended format with examples</li>
                  </ol>
                  <div className="bg-cyan-900/20 border border-cyan-700 p-3 rounded-lg mt-3">
                    <p className="text-cyan-300 text-sm">
                      <strong>Best Practices:</strong> XML tags work best for Claude, Markdown for general use, Pipe format (ID: X | TITLE: Y) for long context.
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Guide 10 - NEW */}
              <AccordionItem value="item-10" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white font-medium flex items-center gap-2">
                    <Wand2 className="h-4 w-4 text-pink-500" />
                    Using the Metaprompt Generator
                    <Badge className="bg-green-600 text-xs">New</Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300 space-y-2 pt-2">
                  <p>Get AI-powered suggestions for minimal prompt improvements:</p>
                  <ol className="list-decimal list-inside space-y-2 mt-2">
                    <li>Click the <strong className="text-pink-400">"Metaprompt"</strong> button from the dashboard</li>
                    <li>Enter your current prompt</li>
                    <li>Describe the <strong>desired behavior</strong> you want</li>
                    <li>Describe the <strong>current undesired behavior</strong></li>
                    <li>Click <strong>"Generate Improvements"</strong></li>
                    <li>Review suggested minimal edits with rationale</li>
                    <li>See the full improved prompt</li>
                    <li>Copy and test the improvements</li>
                  </ol>
                  <div className="bg-pink-900/20 border border-pink-700 p-3 rounded-lg mt-3">
                    <p className="text-pink-300 text-sm">
                      <strong>GPT-5 Metaprompting:</strong> This uses GPT-5's ability to suggest minimal, targeted edits rather than complete rewrites. More precise than traditional rewriting.
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </CardContent>
        </Card>

        {/* FAQ */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">Frequently Asked Questions</CardTitle>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible className="space-y-2">
              <AccordionItem value="faq-1" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">What are the 50 criteria?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="mb-2">The evaluation covers:</p>
                  <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                    <li>Clarity & Specificity</li>
                    <li>Context & Background</li>
                    <li>Task Definition</li>
                    <li>Output Format Requirements</li>
                    <li>Role/Persona Usage</li>
                    <li>Hallucination Minimization</li>
                    <li>Ethical Alignment</li>
                    <li>...and 28 more expert dimensions</li>
                  </ul>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq-2" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">Which LLM provider should I use?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="mb-2">All three providers work excellently:</p>
                  <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                    <li><strong>OpenAI (GPT-4o):</strong> Great balance of quality and speed</li>
                    <li><strong>Claude (Sonnet 4):</strong> Best for detailed, nuanced analysis</li>
                    <li><strong>Gemini (2.0):</strong> Most cost-effective, very fast</li>
                  </ul>
                  <p className="mt-2 text-sm">Choose based on your budget and quality preferences.</p>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq-3" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">How accurate is the cost calculator?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="text-sm">
                    The cost calculator provides estimates based on token counting (1 token ≈ 4 characters) and current API pricing.
                    Actual costs may vary slightly but are typically within 5-10% of the estimate. Use it as a guideline for budgeting.
                  </p>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq-4" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">Can I use custom models?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="text-sm">
                    Yes! In the Settings dialog, you can specify any model name supported by your chosen provider.
                    For example: "gpt-4o-mini", "claude-3-opus-20240229", or "gemini-1.5-pro".
                    Leave blank to use the default recommended model.
                  </p>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq-5" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">How long are evaluations stored?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="text-sm">
                    All evaluations are stored permanently in your MongoDB database until you manually delete them.
                    You can export them as PDF or JSON before deletion to keep records.
                  </p>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq-6" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">What if I get an error?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="mb-2 text-sm">Common issues and solutions:</p>
                  <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                    <li><strong>API Key Error:</strong> Verify your API key is correct and has credits</li>
                    <li><strong>Connection Error:</strong> Check that MongoDB and backend server are running</li>
                    <li><strong>Rate Limit:</strong> Wait a few moments and try again</li>
                    <li><strong>Invalid Model:</strong> Check model name spelling or use default</li>
                    <li><strong>Timeout Error:</strong> Request took too long (Evaluation: 2.5 min, Playground: 1.5 min). Try a shorter prompt or simpler request.</li>
                  </ul>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq-7" className="bg-slate-900 border border-slate-700 rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <span className="text-white">Are there request timeouts?</span>
                </AccordionTrigger>
                <AccordionContent className="text-slate-300">
                  <p className="mb-2 text-sm">Yes, to prevent hanging requests:</p>
                  <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                    <li><strong>Evaluations & Rewrites:</strong> 2 minute backend timeout, 2.5 minute frontend timeout</li>
                    <li><strong>Playground Tests:</strong> 1 minute backend timeout, 1.5 minute frontend timeout</li>
                    <li><strong>Why?</strong> Most prompts complete in 5-30 seconds. Timeouts prevent indefinite waits.</li>
                    <li><strong>If timeout occurs:</strong> Try a shorter/simpler prompt, or wait and retry (might be API congestion)</li>
                  </ul>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </CardContent>
        </Card>

        {/* Tips & Best Practices */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">Tips & Best Practices</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex gap-3 items-start bg-blue-900/20 p-3 rounded-lg border border-blue-700">
                <Lightbulb className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-blue-300 font-semibold text-sm">Iterate for Excellence</h4>
                  <p className="text-slate-300 text-sm">
                    Use the cycle: Evaluate → Rewrite → Evaluate again. Each iteration improves your prompt quality.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 items-start bg-purple-900/20 p-3 rounded-lg border border-purple-700">
                <Lightbulb className="h-5 w-5 text-purple-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-purple-300 font-semibold text-sm">Test Before Evaluating</h4>
                  <p className="text-slate-300 text-sm">
                    Use the Playground to quickly test prompt variations before running full evaluations. Saves time and costs!
                  </p>
                </div>
              </div>

              <div className="flex gap-3 items-start bg-green-900/20 p-3 rounded-lg border border-green-700">
                <Lightbulb className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-green-300 font-semibold text-sm">Optimize Costs</h4>
                  <p className="text-slate-300 text-sm">
                    Use cheaper models like gpt-4o-mini or gemini-2.0-flash-exp for testing and iterations, then switch to premium models for final versions.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 items-start bg-amber-900/20 p-3 rounded-lg border border-amber-700">
                <Lightbulb className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-amber-300 font-semibold text-sm">Focus on Low Scores</h4>
                  <p className="text-slate-300 text-sm">
                    Pay special attention to criteria scored 1-2. These are your biggest improvement opportunities.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 items-start bg-red-900/20 p-3 rounded-lg border border-red-700">
                <Lightbulb className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-red-300 font-semibold text-sm">Save Your Best Prompts</h4>
                  <p className="text-slate-300 text-sm">
                    Export high-scoring prompts as PDF/JSON for future reference and team sharing.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Support */}
        <Card className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 border-blue-700">
          <CardContent className="pt-6">
            <div className="text-center">
              <h3 className="text-white font-bold text-xl mb-2">Need More Help?</h3>
              <p className="text-slate-300 mb-4">
                Check the full documentation or review the API docs
              </p>
              <div className="flex gap-3 justify-center">
                <Button
                  onClick={() => navigate("/")}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  Back to Dashboard
                </Button>
                <Button
                  onClick={() => window.open(`${BASE_URL}/docs`, "_blank")}
                  variant="outline"
                  className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
                >
                  API Documentation
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

