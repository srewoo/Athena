/**
 * Test Results Panel Component
 * Displays test execution results with detailed breakdowns
 */
import React, { useState } from "react";
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Play,
  BarChart3,
  Maximize2
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

const TestResultsPanel = ({
  results,
  summary,
  passThreshold = 3.5,
  onRunSingleTest,
  onRerunFailed,
  isRerunning
}) => {
  const [expandedItems, setExpandedItems] = useState({});
  const [detailViewOpen, setDetailViewOpen] = useState(false);
  const [detailViewItem, setDetailViewItem] = useState(null);

  if (!results || results.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No test results yet. Run a test to see results here.
      </div>
    );
  }

  const toggleExpand = (index) => {
    setExpandedItems(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const openDetailView = (item) => {
    setDetailViewItem(item);
    setDetailViewOpen(true);
  };

  const getVerdictColor = (verdict) => {
    switch (verdict?.toUpperCase()) {
      case "PASS":
        return "text-green-600";
      case "FAIL":
        return "text-red-600";
      case "NEEDS_REVIEW":
        return "text-yellow-600";
      default:
        return "text-muted-foreground";
    }
  };

  const getVerdictIcon = (passed, score) => {
    if (passed || score >= passThreshold) {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    }
    if (score >= passThreshold - 1) {
      return <AlertCircle className="h-5 w-5 text-yellow-500" />;
    }
    return <XCircle className="h-5 w-5 text-red-500" />;
  };

  // Calculate stats
  const passed = results.filter(r => r.passed || r.eval_score >= passThreshold).length;
  const failed = results.length - passed;
  const avgScore = results.reduce((sum, r) => sum + (r.eval_score || 0), 0) / results.length;
  const passRate = (passed / results.length) * 100;

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">{passed}</div>
            <div className="text-sm text-muted-foreground">Passed</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-red-600">{failed}</div>
            <div className="text-sm text-muted-foreground">Failed</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{avgScore.toFixed(1)}</div>
            <div className="text-sm text-muted-foreground">Avg Score</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{passRate.toFixed(0)}%</div>
            <div className="text-sm text-muted-foreground">Pass Rate</div>
          </CardContent>
        </Card>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Overall Pass Rate</span>
          <span className={passRate >= 80 ? "text-green-600" : passRate >= 60 ? "text-yellow-600" : "text-red-600"}>
            {passRate.toFixed(1)}%
          </span>
        </div>
        <Progress value={passRate} className="h-3" />
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {onRunSingleTest && (
          <Button variant="outline" size="sm" onClick={onRunSingleTest}>
            <Play className="h-4 w-4 mr-2" />
            Single Test
          </Button>
        )}
        {failed > 0 && onRerunFailed && (
          <Button variant="outline" size="sm" onClick={onRerunFailed} disabled={isRerunning}>
            <BarChart3 className="h-4 w-4 mr-2" />
            Re-run {failed} Failed
          </Button>
        )}
      </div>

      {/* Results List */}
      <div className="space-y-2">
        <h4 className="font-medium">Test Cases ({results.length})</h4>

        {results.map((result, index) => {
          const isExpanded = expandedItems[index];
          const testCase = result.test_case || {};
          const score = result.eval_score || 0;
          const passed = result.passed || score >= passThreshold;

          return (
            <Card key={index} className={passed ? "" : "border-red-200"}>
              <CardContent className="p-3">
                {/* Header */}
                <div
                  className="flex items-center justify-between cursor-pointer"
                  onClick={() => toggleExpand(index)}
                >
                  <div className="flex items-center gap-3">
                    {getVerdictIcon(passed, score)}
                    <div>
                      <div className="font-medium text-sm">
                        {testCase.category || "Test"} #{index + 1}
                      </div>
                      <div className="text-xs text-muted-foreground truncate max-w-md">
                        {testCase.input?.slice(0, 100) || "No input"}...
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <Badge variant={passed ? "default" : "destructive"}>
                      Score: {score.toFixed(1)}/5
                    </Badge>
                    <Badge variant="outline">
                      {result.latency_ms}ms
                    </Badge>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={(e) => {
                        e.stopPropagation();
                        openDetailView(result);
                      }}
                    >
                      <Maximize2 className="h-4 w-4" />
                    </Button>
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </div>
                </div>

                {/* Expanded Content */}
                {isExpanded && (
                  <div className="mt-4 space-y-4 pt-4 border-t">
                    {/* Input */}
                    <div>
                      <h5 className="text-sm font-medium mb-1">Input</h5>
                      <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-32">
                        {testCase.input || "No input"}
                      </pre>
                    </div>

                    {/* Output */}
                    <div>
                      <h5 className="text-sm font-medium mb-1">Output</h5>
                      <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-48">
                        {result.prompt_output || "No output"}
                      </pre>
                    </div>

                    {/* Feedback */}
                    <div>
                      <h5 className="text-sm font-medium mb-1">Evaluation Feedback</h5>
                      <p className="text-sm text-muted-foreground">
                        {result.eval_feedback || "No feedback"}
                      </p>
                    </div>

                    {/* Dimension Scores */}
                    {result.dimension_scores && Object.keys(result.dimension_scores).length > 0 && (
                      <div>
                        <h5 className="text-sm font-medium mb-1">Dimension Scores</h5>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(result.dimension_scores).map(([dim, data]) => (
                            <div key={dim} className="flex justify-between text-sm p-2 bg-muted/50 rounded">
                              <span className="capitalize">{dim.replace('_', ' ')}</span>
                              <span className="font-medium">
                                {typeof data === 'object' ? data.score : data}/5
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Detail View Dialog */}
      <Dialog open={detailViewOpen} onOpenChange={setDetailViewOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              Test Result Details
            </DialogTitle>
          </DialogHeader>

          {detailViewItem && (
            <div className="space-y-4">
              {/* Score Badge */}
              <div className="flex items-center gap-4">
                {getVerdictIcon(detailViewItem.passed, detailViewItem.eval_score)}
                <Badge variant={detailViewItem.passed ? "default" : "destructive"} className="text-lg px-4 py-1">
                  Score: {detailViewItem.eval_score?.toFixed(2)}/5
                </Badge>
                <Badge variant="outline">
                  Latency: {detailViewItem.latency_ms}ms
                </Badge>
                <Badge variant="outline">
                  Tokens: {detailViewItem.tokens_used}
                </Badge>
              </div>

              {/* Input */}
              <div>
                <h4 className="font-medium mb-2">Input</h4>
                <pre className="text-sm bg-muted p-4 rounded overflow-auto max-h-48 whitespace-pre-wrap">
                  {detailViewItem.test_case?.input || "No input"}
                </pre>
              </div>

              {/* Output */}
              <div>
                <h4 className="font-medium mb-2">System Output</h4>
                <pre className="text-sm bg-muted p-4 rounded overflow-auto max-h-64 whitespace-pre-wrap">
                  {detailViewItem.prompt_output || "No output"}
                </pre>
              </div>

              {/* Feedback */}
              <div>
                <h4 className="font-medium mb-2">Evaluation Feedback</h4>
                <div className="bg-muted p-4 rounded">
                  <p className="text-sm whitespace-pre-wrap">
                    {detailViewItem.eval_feedback || "No feedback"}
                  </p>
                </div>
              </div>

              {/* Dimension Scores */}
              {detailViewItem.dimension_scores && Object.keys(detailViewItem.dimension_scores).length > 0 && (
                <div>
                  <h4 className="font-medium mb-2">Dimension Scores</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(detailViewItem.dimension_scores).map(([dim, data]) => (
                      <Card key={dim}>
                        <CardContent className="p-3">
                          <div className="flex justify-between items-center">
                            <span className="capitalize font-medium">{dim.replace('_', ' ')}</span>
                            <Badge variant={typeof data === 'object' ? (data.score >= 4 ? "default" : data.score >= 3 ? "secondary" : "destructive") : "outline"}>
                              {typeof data === 'object' ? data.score : data}/5
                            </Badge>
                          </div>
                          {typeof data === 'object' && data.rationale && (
                            <p className="text-xs text-muted-foreground mt-1">{data.rationale}</p>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default TestResultsPanel;
