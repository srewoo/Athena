import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Plus, 
  Trash2, 
  Save, 
  RefreshCw,
  Eye,
  EyeOff,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

/**
 * CalibrationExampleEditor Component
 * 
 * Allows users to review and edit calibration examples in eval prompts.
 * Displays score distribution, quality metrics, and allows inline editing.
 */
export const CalibrationExampleEditor = ({
  projectId,
  calibrationExamples = [],
  onSave,
  onClose,
  isOpen = false
}) => {
  const [examples, setExamples] = useState([]);
  const [editingIndex, setEditingIndex] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(true);
  const [activeTab, setActiveTab] = useState('examples');

  useEffect(() => {
    if (calibrationExamples && calibrationExamples.length > 0) {
      setExamples(calibrationExamples.map((ex, idx) => ({ ...ex, id: idx })));
    }
  }, [calibrationExamples]);

  // Calculate statistics
  const stats = React.useMemo(() => {
    if (examples.length === 0) return null;

    const scores = examples.map(ex => ex.score || 3);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

    const scoreDistribution = {
      '5': scores.filter(s => s >= 4.5).length,
      '4': scores.filter(s => s >= 3.5 && s < 4.5).length,
      '3': scores.filter(s => s >= 2.5 && s < 3.5).length,
      '2': scores.filter(s => s >= 1.5 && s < 2.5).length,
      '1': scores.filter(s => s < 1.5).length,
    };

    const hasEdgeCases = examples.filter(ex => ex.is_edge_case).length;
    const hasIntermediateScores = examples.filter(ex => 
      ex.score && !Number.isInteger(ex.score)
    ).length;

    return {
      total: examples.length,
      avgScore: avgScore.toFixed(2),
      scoreDistribution,
      hasEdgeCases,
      hasIntermediateScores,
      coverage: {
        excellent: scoreDistribution['5'],
        good: scoreDistribution['4'],
        acceptable: scoreDistribution['3'],
        poor: scoreDistribution['2'] + scoreDistribution['1']
      }
    };
  }, [examples]);

  const handleEditExample = (index) => {
    setEditingIndex(index);
  };

  const handleUpdateExample = (index, field, value) => {
    const updated = [...examples];
    updated[index] = { ...updated[index], [field]: value };
    setExamples(updated);
  };

  const handleDeleteExample = (index) => {
    setExamples(examples.filter((_, i) => i !== index));
  };

  const handleAddExample = () => {
    const newExample = {
      id: examples.length,
      input: '',
      output: '',
      score: 3,
      reasoning: '',
      dimension_scores: {},
      is_edge_case: false,
      edge_case_type: ''
    };
    setExamples([...examples, newExample]);
    setEditingIndex(examples.length);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSave(examples);
    } finally {
      setIsSaving(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 4.5) return 'text-green-600 bg-green-50';
    if (score >= 3.5) return 'text-blue-600 bg-blue-50';
    if (score >= 2.5) return 'text-yellow-600 bg-yellow-50';
    if (score >= 1.5) return 'text-orange-600 bg-orange-50';
    return 'text-red-600 bg-red-50';
  };

  const getScoreIcon = (score) => {
    if (score >= 4) return <TrendingUp className="h-4 w-4" />;
    if (score >= 3) return <Minus className="h-4 w-4" />;
    return <TrendingDown className="h-4 w-4" />;
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-blue-500" />
            Calibration Examples Editor
          </DialogTitle>
          <DialogDescription>
            Review and edit calibration examples to improve evaluator consistency.
            These examples help LLM judges understand what different scores mean.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="examples">
              Examples ({examples.length})
            </TabsTrigger>
            <TabsTrigger value="statistics">
              Statistics & Coverage
            </TabsTrigger>
          </TabsList>

          {/* Examples Tab */}
          <TabsContent value="examples" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button
                  onClick={handleAddExample}
                  size="sm"
                  variant="outline"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add Example
                </Button>
                <Button
                  onClick={() => setShowPreview(!showPreview)}
                  size="sm"
                  variant="ghost"
                >
                  {showPreview ? (
                    <><Eye className="h-4 w-4 mr-2" /> Preview</>
                  ) : (
                    <><EyeOff className="h-4 w-4 mr-2" /> Edit</>
                  )}
                </Button>
              </div>
              <Badge variant="secondary">
                {examples.length} / 8-10 recommended
              </Badge>
            </div>

            {examples.length === 0 ? (
              <Card>
                <CardContent className="pt-6 text-center text-muted-foreground">
                  <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No calibration examples yet.</p>
                  <p className="text-sm mt-2">Add examples to help calibrate evaluators.</p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-3">
                {examples.map((example, index) => (
                  <Card key={example.id} className="relative">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Badge className={`${getScoreColor(example.score)} border-0`}>
                            {getScoreIcon(example.score)}
                            <span className="ml-1">Score: {example.score}/5</span>
                          </Badge>
                          {example.is_edge_case && (
                            <Badge variant="outline" className="text-orange-600 border-orange-300">
                              <AlertCircle className="h-3 w-3 mr-1" />
                              Edge Case
                            </Badge>
                          )}
                          {example.score && !Number.isInteger(example.score) && (
                            <Badge variant="outline" className="text-purple-600 border-purple-300">
                              Intermediate Score
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={() => handleEditExample(index)}
                            size="sm"
                            variant="ghost"
                          >
                            {editingIndex === index ? 'Cancel' : 'Edit'}
                          </Button>
                          <Button
                            onClick={() => handleDeleteExample(index)}
                            size="sm"
                            variant="ghost"
                            className="text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {editingIndex === index ? (
                        // Edit Mode
                        <div className="space-y-3">
                          <div>
                            <Label>Input</Label>
                            <Textarea
                              value={example.input}
                              onChange={(e) => handleUpdateExample(index, 'input', e.target.value)}
                              rows={2}
                              placeholder="Example input..."
                            />
                          </div>
                          <div>
                            <Label>Output</Label>
                            <Textarea
                              value={example.output}
                              onChange={(e) => handleUpdateExample(index, 'output', e.target.value)}
                              rows={3}
                              placeholder="Example output..."
                            />
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <Label>Score (1-5)</Label>
                              <Input
                                type="number"
                                min="1"
                                max="5"
                                step="0.5"
                                value={example.score}
                                onChange={(e) => handleUpdateExample(index, 'score', parseFloat(e.target.value))}
                              />
                            </div>
                            <div>
                              <Label>Edge Case?</Label>
                              <Select
                                value={example.is_edge_case ? 'yes' : 'no'}
                                onValueChange={(val) => handleUpdateExample(index, 'is_edge_case', val === 'yes')}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="no">No</SelectItem>
                                  <SelectItem value="yes">Yes</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          </div>
                          <div>
                            <Label>Reasoning</Label>
                            <Textarea
                              value={example.reasoning}
                              onChange={(e) => handleUpdateExample(index, 'reasoning', e.target.value)}
                              rows={2}
                              placeholder="Why this score? What makes this example good/bad?"
                            />
                          </div>
                          {example.is_edge_case && (
                            <div>
                              <Label>Edge Case Type</Label>
                              <Input
                                value={example.edge_case_type || ''}
                                onChange={(e) => handleUpdateExample(index, 'edge_case_type', e.target.value)}
                                placeholder="e.g., ambiguous_input, auto_fail_override"
                              />
                            </div>
                          )}
                        </div>
                      ) : (
                        // Preview Mode
                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="font-medium text-muted-foreground">Input:</span>
                            <p className="mt-1 text-foreground line-clamp-2">{example.input || 'No input'}</p>
                          </div>
                          <div>
                            <span className="font-medium text-muted-foreground">Output:</span>
                            <p className="mt-1 text-foreground line-clamp-3">{example.output || 'No output'}</p>
                          </div>
                          <div>
                            <span className="font-medium text-muted-foreground">Reasoning:</span>
                            <p className="mt-1 text-foreground line-clamp-2">{example.reasoning || 'No reasoning'}</p>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Statistics Tab */}
          <TabsContent value="statistics" className="space-y-4">
            {stats && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>Score Distribution</CardTitle>
                    <CardDescription>
                      Ensure balanced representation across score ranges
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(stats.scoreDistribution).map(([score, count]) => {
                        const percentage = (count / stats.total) * 100;
                        return (
                          <div key={score}>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="font-medium">Score {score}</span>
                              <span className="text-muted-foreground">{count} ({percentage.toFixed(0)}%)</span>
                            </div>
                            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${
                                  score === '5' ? 'bg-green-500' :
                                  score === '4' ? 'bg-blue-500' :
                                  score === '3' ? 'bg-yellow-500' :
                                  score === '2' ? 'bg-orange-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Quality Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-2xl font-bold">{stats.total}</div>
                        <div className="text-sm text-muted-foreground">Total Examples</div>
                        <div className="text-xs mt-1 text-muted-foreground">
                          Recommended: 8-10
                        </div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold">{stats.avgScore}</div>
                        <div className="text-sm text-muted-foreground">Average Score</div>
                        <div className="text-xs mt-1 text-muted-foreground">
                          Expected: ~3.0
                        </div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold">{stats.hasEdgeCases}</div>
                        <div className="text-sm text-muted-foreground">Edge Cases</div>
                        <div className="text-xs mt-1 text-muted-foreground">
                          Recommended: 2-3
                        </div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold">{stats.hasIntermediateScores}</div>
                        <div className="text-sm text-muted-foreground">Intermediate Scores</div>
                        <div className="text-xs mt-1 text-muted-foreground">
                          Use 4.5, 3.5, 2.5
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Recommendations</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {stats.total < 8 && (
                      <div className="flex items-start gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-orange-500 mt-0.5" />
                        <span>Add {8 - stats.total} more examples to reach recommended minimum (8)</span>
                      </div>
                    )}
                    {stats.hasEdgeCases === 0 && (
                      <div className="flex items-start gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-orange-500 mt-0.5" />
                        <span>Add 2-3 edge case examples (auto-fail, ambiguous inputs, etc.)</span>
                      </div>
                    )}
                    {stats.hasIntermediateScores === 0 && (
                      <div className="flex items-start gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
                        <span>Consider adding intermediate scores (4.5, 3.5, 2.5) for better calibration</span>
                      </div>
                    )}
                    {stats.coverage.excellent === 0 && (
                      <div className="flex items-start gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-orange-500 mt-0.5" />
                        <span>Add at least one excellent example (score 5)</span>
                      </div>
                    )}
                    {stats.coverage.poor === 0 && (
                      <div className="flex items-start gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-orange-500 mt-0.5" />
                        <span>Add at least one poor example (score 1-2) to show what fails</span>
                      </div>
                    )}
                    {stats.total >= 8 && stats.hasEdgeCases >= 2 && stats.hasIntermediateScores >= 3 && (
                      <div className="flex items-start gap-2 text-sm text-green-600">
                        <CheckCircle className="h-4 w-4 mt-0.5" />
                        <span>Calibration examples meet quality standards!</span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <div className="flex justify-between w-full">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={isSaving}>
              {isSaving ? (
                <><RefreshCw className="h-4 w-4 mr-2 animate-spin" /> Saving...</>
              ) : (
                <><Save className="h-4 w-4 mr-2" /> Save Examples</>
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default CalibrationExampleEditor;
