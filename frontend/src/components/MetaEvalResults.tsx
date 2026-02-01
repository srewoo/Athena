/**
 * MetaEvalResults.tsx
 *
 * Displays meta-evaluation results for eval prompt quality checking
 */

import React, { useState } from 'react';
import './MetaEvalResults.css';

interface AuditScore {
  score: number;
  analysis: string;
}

interface LogicGap {
  gap: string;
  worker_evidence?: string;
  judge_evidence?: string;
}

interface MetaEvalData {
  overall_quality_score: number;
  passes_quality_gate: boolean;
  executive_summary: string;
  audit_breakdown: {
    effectiveness_relevance: AuditScore;
    structural_clarity: AuditScore;
    bias_logic: AuditScore;
    metric_conflation: AuditScore;
    granularity: AuditScore;
  };
  logic_gaps: LogicGap[];
  refinement_roadmap: string[];
  suggested_improvements: Record<string, any>;
}

interface Props {
  result: MetaEvalData;
  onRefine?: () => void;
  onDismiss?: () => void;
  showActions?: boolean;
}

const MetaEvalResults: React.FC<Props> = ({
  result,
  onRefine,
  onDismiss,
  showActions = true
}) => {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const getScoreColor = (score: number): string => {
    if (score >= 8) return '#22c55e'; // Green
    if (score >= 6) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  const getScoreLabel = (score: number): string => {
    if (score >= 8) return 'Excellent';
    if (score >= 6) return 'Good';
    if (score >= 4) return 'Needs Work';
    return 'Poor';
  };

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const auditCategories = [
    {
      key: 'effectiveness_relevance',
      name: 'Effectiveness & Relevance',
      icon: '🎯'
    },
    {
      key: 'structural_clarity',
      name: 'Structural Clarity',
      icon: '🏗️'
    },
    {
      key: 'bias_logic',
      name: 'Bias & Logic',
      icon: '⚖️'
    },
    {
      key: 'metric_conflation',
      name: 'Metric Conflation',
      icon: '🔀'
    },
    {
      key: 'granularity',
      name: 'Granularity',
      icon: '🔬'
    }
  ];

  return (
    <div className="meta-eval-results">
      {/* Overall Score Header */}
      <div className={`overall-score-header ${result.passes_quality_gate ? 'pass' : 'fail'}`}>
        <div className="score-badge">
          <div className="score-circle">
            <span className="score-value">{result.overall_quality_score.toFixed(1)}</span>
            <span className="score-max">/10</span>
          </div>
          <div className="score-label">
            {result.passes_quality_gate ? '✓ Passes Quality Gate' : '⚠ Needs Improvement'}
          </div>
        </div>

        <div className="executive-summary">
          <h4>Executive Summary</h4>
          <p>{result.executive_summary}</p>
        </div>
      </div>

      {/* Audit Breakdown */}
      <div className="audit-breakdown">
        <h3>📊 5-Point Audit</h3>

        <div className="audit-categories">
          {auditCategories.map(category => {
            const audit = result.audit_breakdown[category.key as keyof typeof result.audit_breakdown];
            const isExpanded = expandedSection === category.key;

            return (
              <div key={category.key} className="audit-category">
                <div
                  className="category-header"
                  onClick={() => toggleSection(category.key)}
                >
                  <div className="category-info">
                    <span className="category-icon">{category.icon}</span>
                    <span className="category-name">{category.name}</span>
                  </div>

                  <div className="category-score">
                    <span
                      className="score-badge"
                      style={{ backgroundColor: getScoreColor(audit.score) }}
                    >
                      {audit.score.toFixed(1)}
                    </span>
                    <span className="score-label">{getScoreLabel(audit.score)}</span>
                    <span className="chevron">{isExpanded ? '▼' : '▶'}</span>
                  </div>
                </div>

                {isExpanded && (
                  <div className="category-analysis">
                    <p>{audit.analysis}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Logic Gaps */}
      {result.logic_gaps && result.logic_gaps.length > 0 && (
        <div className="logic-gaps">
          <h3>🔍 Logic Gaps Found</h3>
          <div className="gaps-list">
            {result.logic_gaps.map((gap, idx) => (
              <div key={idx} className="gap-item">
                <div className="gap-header">
                  <span className="gap-number">{idx + 1}</span>
                  <span className="gap-description">{gap.gap}</span>
                </div>

                {(gap.worker_evidence || gap.judge_evidence) && (
                  <div className="gap-evidence">
                    {gap.worker_evidence && (
                      <div className="evidence-item">
                        <strong>System Prompt expects:</strong>
                        <code>{gap.worker_evidence}</code>
                      </div>
                    )}
                    {gap.judge_evidence && (
                      <div className="evidence-item">
                        <strong>Eval Prompt has:</strong>
                        <code>{gap.judge_evidence}</code>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Refinement Roadmap */}
      {result.refinement_roadmap && result.refinement_roadmap.length > 0 && (
        <div className="refinement-roadmap">
          <h3>🛠️ Refinement Roadmap</h3>
          <ol className="roadmap-list">
            {result.refinement_roadmap.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Suggested Improvements */}
      {result.suggested_improvements && Object.keys(result.suggested_improvements).length > 0 && (
        <div className="suggested-improvements">
          <h3>💡 Suggested Improvements</h3>
          <div className="improvements-grid">
            {Object.entries(result.suggested_improvements).map(([key, value]) => {
              if (!value || (Array.isArray(value) && value.length === 0)) return null;

              return (
                <div key={key} className="improvement-item">
                  <strong>{key.replace(/_/g, ' ')}:</strong>
                  {Array.isArray(value) ? (
                    <ul>
                      {value.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    <span>{value}</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Actions */}
      {showActions && (
        <div className="meta-eval-actions">
          {onDismiss && (
            <button className="btn-secondary" onClick={onDismiss}>
              Dismiss
            </button>
          )}
          {onRefine && !result.passes_quality_gate && (
            <button className="btn-primary" onClick={onRefine}>
              🔄 Auto-Refine Eval Prompt
            </button>
          )}
          {result.passes_quality_gate && (
            <button className="btn-success" disabled>
              ✓ Quality Gate Passed
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default MetaEvalResults;
