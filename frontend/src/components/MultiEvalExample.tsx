/**
 * MultiEvalExample.tsx
 *
 * Complete example showing how to use the multi-evaluator system
 * in your application
 */

import React, { useState, useEffect } from 'react';
import useMultiEvaluator from '../hooks/useMultiEvaluator';
import MultiEvaluatorResults from './MultiEvaluatorResults';
import DimensionPreview from './DimensionPreview';
import './MultiEvalExample.css';

interface Props {
  projectId: string;
}

const MultiEvalExample: React.FC<Props> = ({ projectId }) => {
  const {
    evaluate,
    detectDimensions,
    compare,
    loading,
    error
  } = useMultiEvaluator(projectId);

  const [inputData, setInputData] = useState('');
  const [output, setOutput] = useState('');
  const [result, setResult] = useState<any>(null);
  const [autoDetect, setAutoDetect] = useState<any>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [mode, setMode] = useState<'evaluate' | 'compare'>('evaluate');

  // Auto-detect dimensions when component mounts
  useEffect(() => {
    loadDimensionPreview();
  }, [projectId]);

  const loadDimensionPreview = async () => {
    const detected = await detectDimensions();
    if (detected) {
      setAutoDetect(detected);
    }
  };

  const handleEvaluate = async () => {
    if (mode === 'compare') {
      const compResult = await compare(inputData, output);
      if (compResult) {
        setResult(compResult.multi_evaluator);
        setComparisonData(compResult.comparison);
      }
    } else {
      const evalResult = await evaluate(inputData, output);
      if (evalResult) {
        setResult(evalResult);
        setComparisonData(null);
      }
    }
  };

  const handlePreviewConfirm = () => {
    setShowPreview(false);
    handleEvaluate();
  };

  return (
    <div className="multi-eval-example">
      <div className="eval-header">
        <h2>🚀 Multi-Evaluator System</h2>
        <p className="header-subtitle">
          Automatic dimension detection • 70% cheaper • 3x faster
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={mode === 'evaluate' ? 'active' : ''}
          onClick={() => setMode('evaluate')}
        >
          Evaluate
        </button>
        <button
          className={mode === 'compare' ? 'active' : ''}
          onClick={() => setMode('compare')}
        >
          Compare (Multi vs Mono)
        </button>
      </div>

      {/* Auto-Detect Summary */}
      {autoDetect && !showPreview && (
        <div className="auto-detect-summary">
          <div className="summary-header">
            <span className="summary-title">
              ✨ Auto-Detected Plan
            </span>
            <button
              className="view-details-btn"
              onClick={() => setShowPreview(true)}
            >
              View Details
            </button>
          </div>
          <div className="summary-stats">
            <span>{autoDetect.total_evaluators} evaluators</span>
            <span>•</span>
            <span>${autoDetect.estimated_cost.toFixed(3)} estimated</span>
            <span>•</span>
            <span>{(autoDetect.estimated_latency_ms / 1000).toFixed(1)}s</span>
          </div>
        </div>
      )}

      {/* Dimension Preview Modal */}
      {showPreview && autoDetect && (
        <div className="modal-overlay" onClick={() => setShowPreview(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <DimensionPreview
              autoDetectResult={autoDetect}
              onConfirm={handlePreviewConfirm}
              onCancel={() => setShowPreview(false)}
            />
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className="eval-form">
        <div className="form-group">
          <label htmlFor="input-data">Input Data</label>
          <textarea
            id="input-data"
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder="Enter the input that was sent to your system..."
            rows={4}
          />
        </div>

        <div className="form-group">
          <label htmlFor="output">System Output</label>
          <textarea
            id="output"
            value={output}
            onChange={(e) => setOutput(e.target.value)}
            placeholder="Enter the output from your system to evaluate..."
            rows={6}
          />
        </div>

        <button
          className="evaluate-btn"
          onClick={handleEvaluate}
          disabled={loading || !inputData || !output}
        >
          {loading ? (
            <>
              <span className="spinner">⏳</span>
              {mode === 'compare' ? 'Comparing...' : 'Evaluating...'}
            </>
          ) : (
            <>
              ▶ {mode === 'compare' ? 'Compare Both' : 'Run Evaluation'}
            </>
          )}
        </button>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="results-section">
          <MultiEvaluatorResults
            result={result}
            showComparison={mode === 'compare'}
            comparisonData={comparisonData}
          />
        </div>
      )}

      {/* Help Text */}
      <div className="help-text">
        <h4>💡 How it works:</h4>
        <ol>
          <li>
            <strong>Auto-Detection:</strong> System analyzes your prompt and
            automatically detects which dimensions to evaluate
          </li>
          <li>
            <strong>Tier 1 (Fast):</strong> Quick auto-fail checks run first
            in parallel (schema, safety, completeness)
          </li>
          <li>
            <strong>Tier 2 (Deep):</strong> Quality evaluations run in parallel
            if Tier 1 passes (accuracy, relevance, actionability, tone)
          </li>
          <li>
            <strong>Aggregation:</strong> Weighted scores combine into final
            verdict with per-dimension attribution
          </li>
        </ol>
      </div>
    </div>
  );
};

export default MultiEvalExample;
