/**
 * DimensionPreview.tsx
 *
 * Shows which dimensions will be evaluated before running evaluation
 * (preview of auto-detection results)
 */

import React from 'react';
import './DimensionPreview.css';

interface DetectedDimension {
  name: string;
  evaluator_type: string;
  weight: number;
  is_critical: boolean;
  min_pass_score: number;
  detection_reason: string;
}

interface AutoDetectResult {
  dimensions: DetectedDimension[];
  total_evaluators: number;
  tier1_count: number;
  tier2_count: number;
  estimated_cost: number;
  estimated_latency_ms: number;
}

interface Props {
  autoDetectResult: AutoDetectResult;
  onConfirm?: () => void;
  onCancel?: () => void;
}

const DimensionPreview: React.FC<Props> = ({
  autoDetectResult,
  onConfirm,
  onCancel
}) => {
  const getTierLabel = (evaluatorType: string): string => {
    if (evaluatorType.includes('Validator') || evaluatorType.includes('Checker')) {
      return 'Tier 1 (Auto-Fail)';
    }
    return 'Tier 2 (Quality)';
  };

  const getTierClass = (evaluatorType: string): string => {
    if (evaluatorType.includes('Validator') || evaluatorType.includes('Checker')) {
      return 'tier-1';
    }
    return 'tier-2';
  };

  return (
    <div className="dimension-preview">
      <div className="preview-header">
        <h3>📊 Evaluation Plan</h3>
        <p className="subtitle">
          Auto-detected {autoDetectResult.total_evaluators} dimensions to evaluate
        </p>
      </div>

      {/* Summary Stats */}
      <div className="preview-stats">
        <div className="stat-card">
          <span className="stat-label">Total Evaluators</span>
          <span className="stat-value">{autoDetectResult.total_evaluators}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Tier 1 (Fast)</span>
          <span className="stat-value tier-1-color">
            {autoDetectResult.tier1_count}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Tier 2 (Deep)</span>
          <span className="stat-value tier-2-color">
            {autoDetectResult.tier2_count}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Est. Cost</span>
          <span className="stat-value">${autoDetectResult.estimated_cost.toFixed(3)}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Est. Time</span>
          <span className="stat-value">
            {(autoDetectResult.estimated_latency_ms / 1000).toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Dimension List */}
      <div className="dimensions-list">
        <h4>Dimensions to Evaluate:</h4>

        {autoDetectResult.dimensions.map((dimension, idx) => (
          <div
            key={idx}
            className={`dimension-item ${getTierClass(dimension.evaluator_type)}`}
          >
            <div className="dimension-header">
              <div className="dimension-info">
                <span className="dimension-number">{idx + 1}</span>
                <span className="dimension-name">
                  {dimension.name.replace(/_/g, ' ')}
                </span>
                {dimension.is_critical && (
                  <span className="critical-badge">⚠ Critical</span>
                )}
              </div>
              <div className="dimension-stats">
                <span className="weight">{(dimension.weight * 100).toFixed(0)}%</span>
                <span className={`tier-badge ${getTierClass(dimension.evaluator_type)}`}>
                  {getTierLabel(dimension.evaluator_type)}
                </span>
              </div>
            </div>

            <div className="dimension-details">
              <p className="detection-reason">
                <strong>Why:</strong> {dimension.detection_reason}
              </p>
              <div className="dimension-meta">
                <span>Min Score: {dimension.min_pass_score}/5.0</span>
                <span>Evaluator: {dimension.evaluator_type}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Execution Flow */}
      <div className="execution-flow">
        <h4>Execution Flow:</h4>
        <div className="flow-steps">
          <div className="flow-step">
            <span className="step-number">1</span>
            <div className="step-content">
              <strong>Run Tier 1 Checks (Parallel)</strong>
              <p>{autoDetectResult.tier1_count} auto-fail checks execute simultaneously</p>
            </div>
          </div>
          <div className="flow-arrow">↓</div>
          <div className="flow-step">
            <span className="step-number">2</span>
            <div className="step-content">
              <strong>If All Pass → Run Tier 2 (Parallel)</strong>
              <p>{autoDetectResult.tier2_count} quality evaluators execute simultaneously</p>
            </div>
          </div>
          <div className="flow-arrow">↓</div>
          <div className="flow-step">
            <span className="step-number">3</span>
            <div className="step-content">
              <strong>Aggregate Results</strong>
              <p>Weighted average + dimension checks → Final verdict</p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      {(onConfirm || onCancel) && (
        <div className="preview-actions">
          {onCancel && (
            <button className="btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          )}
          {onConfirm && (
            <button className="btn-primary" onClick={onConfirm}>
              ✓ Run Evaluation
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default DimensionPreview;
