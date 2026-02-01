/**
 * SeparateEvalPrompts.tsx
 *
 * Displays multiple separate eval prompts (one per dimension)
 * each in its own box with meta-evaluation results
 */

import React, { useState, useEffect } from 'react';
import './SeparateEvalPrompts.css';

interface MetaEvaluation {
  quality_score: number;
  passes_gate: boolean;
  executive_summary: string;
  audit_scores: {
    effectiveness: number;
    structural_clarity: number;
    bias: number;
    metric_conflation: number;
    granularity: number;
  };
  logic_gaps: Array<{
    gap: string;
    worker_evidence?: string;
    judge_evidence?: string;
  }>;
  refinement_roadmap: string[];
}

interface EvalPromptData {
  dimension_name: string;
  evaluator_type: string;
  weight: number;
  is_critical: boolean;
  eval_prompt: string;
  meta_evaluation: MetaEvaluation;
  was_refined: boolean;
  original_prompt?: string;
}

interface Props {
  projectId: string;
}

const SeparateEvalPrompts: React.FC<Props> = ({ projectId }) => {
  const [evaluators, setEvaluators] = useState<EvalPromptData[]>([]);
  const [extractionMetadata, setExtractionMetadata] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedEval, setExpandedEval] = useState<string | null>(null);
  const [showMetaDetails, setShowMetaDetails] = useState<string | null>(null);

  useEffect(() => {
    loadSeparateEvalPrompts();
  }, [projectId]);

  const loadSeparateEvalPrompts = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/projects/${projectId}/multi-eval/separate-prompts`
      );

      if (!response.ok) {
        throw new Error('Failed to load eval prompts');
      }

      const data = await response.json();
      setEvaluators(data.evaluators || []);
      setExtractionMetadata(data.extraction_metadata || null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getQualityColor = (score: number): string => {
    if (score >= 8) return '#22c55e'; // Green
    if (score >= 6) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  const getQualityLabel = (score: number): string => {
    if (score >= 8) return 'Excellent';
    if (score >= 6) return 'Good';
    if (score >= 4) return 'Needs Work';
    return 'Poor';
  };

  if (loading) {
    return <div className="loading">Loading eval prompts...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="separate-eval-prompts">
      <div className="header">
        <h2>📊 Evaluation Prompts (Multi-Evaluator System)</h2>
        <p className="subtitle">
          Each dimension has its own dedicated eval prompt, meta-evaluated for quality
        </p>
        <div className="stats">
          <span className="stat">
            <strong>{evaluators.length}</strong> evaluators
          </span>
          <span className="stat">
            <strong>{evaluators.filter(e => e.meta_evaluation.passes_gate).length}</strong> pass quality gate
          </span>
          <span className="stat">
            <strong>{evaluators.filter(e => e.was_refined).length}</strong> auto-refined
          </span>
        </div>

        {/* Extraction Metadata */}
        {extractionMetadata && (
          <div style={{
            marginTop: '20px',
            padding: '16px',
            background: extractionMetadata.method === 'intelligent_llm_extraction' ? '#e0f2fe' : '#fef3c7',
            borderRadius: '8px',
            border: `2px solid ${extractionMetadata.method === 'intelligent_llm_extraction' ? '#0ea5e9' : '#f59e0b'}`
          }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: 'bold' }}>
              {extractionMetadata.method === 'intelligent_llm_extraction' ? '🧠 Intelligent Extraction Used' : '📝 Regex Detection Used'}
            </h3>
            {extractionMetadata.method === 'intelligent_llm_extraction' ? (
              <div style={{ fontSize: '13px' }}>
                <div style={{ marginBottom: '8px' }}>
                  <strong>Domain:</strong> {extractionMetadata.domain || 'general'} |{' '}
                  <strong>Risk Level:</strong> {extractionMetadata.risk_level || 'medium'} |{' '}
                  <strong>Confidence:</strong> {((extractionMetadata.confidence_score || 0) * 100).toFixed(0)}%
                </div>
                {extractionMetadata.critical_dimensions && extractionMetadata.critical_dimensions.length > 0 && (
                  <div style={{ marginBottom: '8px' }}>
                    <strong>Critical Dimensions:</strong> {extractionMetadata.critical_dimensions.join(', ')}
                  </div>
                )}
                {extractionMetadata.quality_priorities && extractionMetadata.quality_priorities.length > 0 && (
                  <div>
                    <strong>Quality Priorities:</strong> {extractionMetadata.quality_priorities.slice(0, 3).join(', ')}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: '13px' }}>
                Using legacy regex-based pattern matching.
              </div>
            )}
          </div>
        )}
      </div>

      <div className="evaluators-grid">
        {evaluators.map((evaluator) => (
          <EvalPromptBox
            key={evaluator.dimension_name}
            evaluator={evaluator}
            isExpanded={expandedEval === evaluator.dimension_name}
            showMetaDetails={showMetaDetails === evaluator.dimension_name}
            onToggleExpand={() => setExpandedEval(
              expandedEval === evaluator.dimension_name ? null : evaluator.dimension_name
            )}
            onToggleMetaDetails={() => setShowMetaDetails(
              showMetaDetails === evaluator.dimension_name ? null : evaluator.dimension_name
            )}
            getQualityColor={getQualityColor}
            getQualityLabel={getQualityLabel}
          />
        ))}
      </div>
    </div>
  );
};

interface EvalPromptBoxProps {
  evaluator: EvalPromptData;
  isExpanded: boolean;
  showMetaDetails: boolean;
  onToggleExpand: () => void;
  onToggleMetaDetails: () => void;
  getQualityColor: (score: number) => string;
  getQualityLabel: (score: number) => string;
}

const EvalPromptBox: React.FC<EvalPromptBoxProps> = ({
  evaluator,
  isExpanded,
  showMetaDetails,
  onToggleExpand,
  onToggleMetaDetails,
  getQualityColor,
  getQualityLabel
}) => {
  const qualityScore = evaluator.meta_evaluation.quality_score;
  const qualityColor = getQualityColor(qualityScore);

  return (
    <div className={`eval-prompt-box ${evaluator.meta_evaluation.passes_gate ? 'passes' : 'needs-work'}`}>
      {/* Header */}
      <div className="box-header">
        <div className="header-left">
          <h3 className="dimension-name">
            {evaluator.dimension_name.replace(/_/g, ' ')}
          </h3>
          {evaluator.is_critical && (
            <span className="critical-badge">⚠ Critical</span>
          )}
          {evaluator.was_refined && (
            <span className="refined-badge">🔄 Auto-Refined</span>
          )}
        </div>
        <div className="header-right">
          <div className="quality-badge" style={{ backgroundColor: qualityColor }}>
            <span className="quality-score">{qualityScore.toFixed(1)}</span>
            <span className="quality-label">{getQualityLabel(qualityScore)}</span>
          </div>
        </div>
      </div>

      {/* Meta Info */}
      <div className="meta-info">
        <span className="info-item">Weight: {(evaluator.weight * 100).toFixed(0)}%</span>
        <span className="info-item">Type: {evaluator.evaluator_type}</span>
        <span className="info-item">
          {evaluator.meta_evaluation.passes_gate ? '✓ Passes Gate' : '⚠ Needs Improvement'}
        </span>
      </div>

      {/* Executive Summary */}
      <div className="executive-summary">
        <p>{evaluator.meta_evaluation.executive_summary}</p>
      </div>

      {/* Action Buttons */}
      <div className="action-buttons">
        <button
          className="btn-view-prompt"
          onClick={onToggleExpand}
        >
          {isExpanded ? '▼ Hide' : '▶ View'} Eval Prompt
        </button>
        <button
          className="btn-meta-details"
          onClick={onToggleMetaDetails}
        >
          📊 Meta-Eval Details
        </button>
      </div>

      {/* Expanded: Eval Prompt */}
      {isExpanded && (
        <div className="eval-prompt-content">
          <h4>Evaluation Prompt:</h4>
          <pre className="prompt-text">{evaluator.eval_prompt}</pre>
        </div>
      )}

      {/* Expanded: Meta Details */}
      {showMetaDetails && (
        <div className="meta-details">
          <h4>Meta-Evaluation Audit:</h4>

          <div className="audit-scores">
            {Object.entries(evaluator.meta_evaluation.audit_scores).map(([key, score]) => (
              <div key={key} className="audit-score-item">
                <span className="audit-label">{key.replace(/_/g, ' ')}:</span>
                <div className="audit-bar">
                  <div
                    className="audit-fill"
                    style={{
                      width: `${(score / 10) * 100}%`,
                      backgroundColor: getQualityColor(score)
                    }}
                  />
                </div>
                <span className="audit-value">{score.toFixed(1)}/10</span>
              </div>
            ))}
          </div>

          {evaluator.meta_evaluation.logic_gaps.length > 0 && (
            <div className="logic-gaps">
              <h5>Logic Gaps:</h5>
              <ul>
                {evaluator.meta_evaluation.logic_gaps.map((gap, idx) => (
                  <li key={idx}>{gap.gap}</li>
                ))}
              </ul>
            </div>
          )}

          {evaluator.meta_evaluation.refinement_roadmap.length > 0 && (
            <div className="refinement-roadmap">
              <h5>Refinement Roadmap:</h5>
              <ol>
                {evaluator.meta_evaluation.refinement_roadmap.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SeparateEvalPrompts;
