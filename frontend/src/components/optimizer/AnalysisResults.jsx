/**
 * Analysis Results Component
 * Displays prompt analysis including DNA, quality scores, and suggestions
 */
import React from "react";
import { TrendingUp, TrendingDown, Minus, AlertTriangle, CheckCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";

const AnalysisResults = ({ analysis }) => {
  if (!analysis) return null;

  const {
    prompt_type,
    prompt_types_detected = [],
    quality_score = 0,
    programmatic_score = null,
    llm_score = null,
    combined_score = null,
    llm_enhanced = false,
    quality_breakdown = {},
    improvement_needed = false,
    improvement_areas = [],
    strengths = [],
    dna = {},
    suggested_eval_dimensions = [],
    suggested_test_categories = []
  } = analysis;

  // Use combined_score if available, otherwise fall back to quality_score
  const displayScore = combined_score || quality_score;
  const hasMultipleScores = programmatic_score !== null && llm_score !== null && llm_score > 0;

  const getScoreColor = (score) => {
    if (score >= 8) return "text-green-600";
    if (score >= 6) return "text-yellow-600";
    return "text-red-600";
  };

  const getScoreIcon = (score) => {
    if (score >= 8) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (score >= 6) return <Minus className="h-4 w-4 text-yellow-600" />;
    return <TrendingDown className="h-4 w-4 text-red-600" />;
  };

  return (
    <div className="space-y-4">
      {/* Overall Score */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Quality Score</CardTitle>
            <div className={`text-3xl font-bold ${getScoreColor(displayScore)}`}>
              {displayScore}/10
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Progress value={displayScore * 10} className="h-2" />

          {/* Dual Score Display - Show both programmatic and LLM scores */}
          {hasMultipleScores && (
            <div className="mt-4 p-3 bg-muted/50 rounded-lg">
              <div className="text-xs font-medium text-muted-foreground mb-2">Score Breakdown</div>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-2 bg-background rounded border">
                  <div className="text-xs text-muted-foreground">Pattern Analysis</div>
                  <div className={`text-xl font-bold ${getScoreColor(programmatic_score)}`}>
                    {programmatic_score?.toFixed(1)}/10
                  </div>
                  <div className="text-xs text-muted-foreground">Rule-based (30%)</div>
                </div>
                <div className="text-center p-2 bg-background rounded border">
                  <div className="text-xs text-muted-foreground">LLM Analysis</div>
                  <div className={`text-xl font-bold ${getScoreColor(llm_score)}`}>
                    {llm_score?.toFixed(1)}/10
                  </div>
                  <div className="text-xs text-muted-foreground">Semantic (70%)</div>
                </div>
              </div>
              {Math.abs(programmatic_score - llm_score) > 1.5 && (
                <div className="mt-2 text-xs text-yellow-600 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  Score discrepancy: Pattern analysis finds issues that LLM considers acceptable
                </div>
              )}
            </div>
          )}

          {/* Show if LLM analysis was not available */}
          {!llm_enhanced && (
            <div className="mt-3 text-xs text-muted-foreground bg-yellow-50 dark:bg-yellow-900/20 p-2 rounded">
              ⚠️ LLM analysis unavailable. Score based on pattern matching only.
            </div>
          )}

          {/* Breakdown */}
          <div className="grid grid-cols-2 gap-4 mt-4">
            {Object.entries(quality_breakdown).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-sm capitalize">{key.replace('_', ' ')}</span>
                <div className="flex items-center gap-2">
                  {getScoreIcon(value)}
                  <span className={`font-medium ${getScoreColor(value)}`}>
                    {value}/10
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Prompt Type */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Prompt Type</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Badge variant="default" className="capitalize">
              {prompt_type?.replace('_', ' ') || 'Unknown'}
            </Badge>
            {prompt_types_detected?.slice(1).map((type, i) => (
              <Badge key={i} variant="outline" className="capitalize">
                {type?.replace('_', ' ')}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* DNA Elements */}
      {dna && Object.keys(dna).length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Prompt DNA</CardTitle>
            <CardDescription>Core elements that will be preserved</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {dna.role && (
              <div>
                <span className="text-sm font-medium">Role:</span>
                <p className="text-sm text-muted-foreground">{dna.role}</p>
              </div>
            )}

            {dna.output_format && (
              <div>
                <span className="text-sm font-medium">Output Format:</span>
                <Badge variant="secondary" className="ml-2">{dna.output_format}</Badge>
              </div>
            )}

            {dna.template_variables?.length > 0 && (
              <div>
                <span className="text-sm font-medium">Template Variables:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {dna.template_variables.map((v, i) => (
                    <Badge key={i} variant="outline" className="font-mono text-xs">
                      {`{{${v}}}`}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {dna.scoring_scale && (
              <div>
                <span className="text-sm font-medium">Scoring Scale:</span>
                <span className="text-sm text-muted-foreground ml-2">
                  {dna.scoring_scale.min}-{dna.scoring_scale.max} ({dna.scoring_scale.type})
                </span>
              </div>
            )}

            {dna.sections?.length > 0 && (
              <div>
                <span className="text-sm font-medium">Sections:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {dna.sections.slice(0, 5).map((s, i) => (
                    <Badge key={i} variant="outline" className="text-xs">
                      {s}
                    </Badge>
                  ))}
                  {dna.sections.length > 5 && (
                    <Badge variant="outline" className="text-xs">
                      +{dna.sections.length - 5} more
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Strengths & Improvements */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Strengths */}
        {strengths?.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                Strengths
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {strengths.map((s, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <span className="text-green-500 mt-1">+</span>
                    {s}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Improvement Areas */}
        {improvement_areas?.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-yellow-500" />
                Areas to Improve
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {improvement_areas.map((area, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <span className="text-yellow-500 mt-1">!</span>
                    {area}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Suggested Eval Dimensions */}
      {suggested_eval_dimensions?.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Suggested Evaluation Dimensions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {suggested_eval_dimensions.map((dim, i) => (
                <div key={i} className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-sm">{dim.name}</div>
                  <div className="text-xs text-muted-foreground">{dim.description}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AnalysisResults;
