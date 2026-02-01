/**
 * MultiEvaluatorResults.tsx
 *
 * Displays results from the multi-evaluator system with
 * dimension-by-dimension breakdown
 */

import React, { useState } from 'react';
import './MultiEvaluatorResults.css';

interface EvaluationDimension {
  dimension_name: string;
  score: number;
  passes: boolean;
  reason: string;
  weight: number;
  is_critical: boolean;
  evidence?: string[];
  model_used?: string;
  latency_ms?: number;
}

interface MultiEvalResult {
  verdict: string;
  score: number;
  reason: string;
  dimension_scores: Record<string, number>;
  individual_evaluations: EvaluationDimension[];
  performance: {
    total_latency_ms: number;
    total_tokens_used: number;
    auto_fail_triggered?: boolean;
  };
  evaluator_type?: string;
}

interface Props {
  result: MultiEvalResult;
  showComparison?: boolean;
  comparisonData?: {
    cost_savings_pct?: number;
    latency_improvement_pct?: number;
  };
}

const MultiEvaluatorResults: React.FC<Props> = ({
  result,
  showComparison = false,
  comparisonData
}) => {
  const [expandedDimension, setExpandedDimension] = useState<string | null>(null);

  const getVerdictClass = (verdict: string) => {
    switch (verdict) {
      case 'PASS': return 'verdict-pass';
      case 'FAIL': return 'verdict-fail';
      case 'NEEDS_REVIEW': return 'verdict-review';
      default: return '';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 4.5) return '#22c55e'; // Green
    if (score >= 4.0) return '#84cc16'; // Light green
    if (score >= 3.0) return '#f59e0b'; // Yellow
    if (score >= 2.0) return '#f97316'; // Orange
    return '#ef4444'; // Red
  };

  const toggleDimension = (dimensionName: string) => {
    setExpandedDimension(
      expandedDimension === dimensionName ? null : dimensionName
    );
  };

  return (
    <div className="multi-eval-results">
      {/* Overall Verdict Header */}
      <div className={`overall-verdict ${getVerdictClass(result.verdict)}`}>
        <div className="verdict-badge">
          {result.verdict === 'PASS' && '✓'}
          {result.verdict === 'FAIL' && '✗'}
          {result.verdict === 'NEEDS_REVIEW' && '⚠'}
          <span className="verdict-text">{result.verdict}</span>
        </div>
        <div className="overall-score">
          <span className="score-value">{result.score.toFixed(1)}</span>
          <span className="score-max">/5.0</span>
        </div>
      </div>

      {/* Overall Reason */}
      <div className="overall-reason">
        <p>{result.reason}</p>
      </div>

      {/* Auto-Fail Alert */}
      {result.performance.auto_fail_triggered && (
        <div className="auto-fail-alert">
          <span className="alert-icon">⚠️</span>
          <span>Auto-fail condition triggered</span>
        </div>
      )}

      {/* Comparison Metrics (if shown) */}
      {showComparison && comparisonData && (
        <div className="comparison-metrics">
          <div className="metric">
            <span className="metric-label">Cost Savings:</span>
            <span className="metric-value success">
              {comparisonData.cost_savings_pct?.toFixed(0)}%
            </span>
          </div>
          <div className="metric">
            <span className="metric-label">Speed Improvement:</span>
            <span className="metric-value success">
              {comparisonData.latency_improvement_pct?.toFixed(0)}%
            </span>
          </div>
        </div>
      )}

      {/* Dimension Cards Grid */}
      <div className="dimensions-grid">
        {result.individual_evaluations.map((dimension) => (
          <DimensionCard
            key={dimension.dimension_name}
            dimension={dimension}
            isExpanded={expandedDimension === dimension.dimension_name}
            onToggle={() => toggleDimension(dimension.dimension_name)}
            scoreColor={getScoreColor(dimension.score)}
          />
        ))}
      </div>

      {/* Performance Footer */}
      <div className="performance-footer">
        <div className="perf-stat">
          <span className="perf-label">Latency:</span>
          <span className="perf-value">
            {result.performance.total_latency_ms}ms
          </span>
        </div>
        <div className="perf-stat">
          <span className="perf-label">Tokens:</span>
          <span className="perf-value">
            {result.performance.total_tokens_used.toLocaleString()}
          </span>
        </div>
        <div className="perf-stat">
          <span className="perf-label">Type:</span>
          <span className="perf-value">
            {result.evaluator_type || 'multi-evaluator'}
          </span>
        </div>
      </div>
    </div>
  );
};

interface DimensionCardProps {
  dimension: EvaluationDimension;
  isExpanded: boolean;
  onToggle: () => void;
  scoreColor: string;
}

const DimensionCard: React.FC<DimensionCardProps> = ({
  dimension,
  isExpanded,
  onToggle,
  scoreColor
}) => {
  const statusIcon = dimension.passes ? '✅' : '❌';
  const cardClass = `dimension-card ${dimension.passes ? 'pass' : 'fail'}`;

  return (
    <div className={cardClass}>
      <div className="card-header" onClick={onToggle}>
        <div className="header-left">
          <span className="status-icon">{statusIcon}</span>
          <span className="dimension-name">
            {dimension.dimension_name.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="header-right">
          <span className="dimension-score" style={{ color: scoreColor }}>
            {dimension.score.toFixed(1)}
          </span>
          <span className="chevron">{isExpanded ? '▼' : '▶'}</span>
        </div>
      </div>

      <div className="card-meta">
        <span className="weight">
          Weight: {(dimension.weight * 100).toFixed(0)}%
        </span>
        {dimension.is_critical && (
          <span className="badge critical">Critical</span>
        )}
      </div>

      {/* Progress Bar */}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${(dimension.score / 5) * 100}%`,
            backgroundColor: scoreColor
          }}
        />
      </div>

      {/* Reason (always visible, truncated) */}
      <p className={`reason ${isExpanded ? 'expanded' : 'truncated'}`}>
        {dimension.reason}
      </p>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="expanded-details">
          {dimension.evidence && dimension.evidence.length > 0 && (
            <div className="evidence-section">
              <h4>Evidence:</h4>
              <ul>
                {dimension.evidence.map((ev, idx) => (
                  <li key={idx}>{ev}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="dimension-meta">
            {dimension.model_used && (
              <span className="meta-item">
                Model: {dimension.model_used}
              </span>
            )}
            {dimension.latency_ms && (
              <span className="meta-item">
                {dimension.latency_ms}ms
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiEvaluatorResults;
