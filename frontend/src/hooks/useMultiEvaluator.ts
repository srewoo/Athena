/**
 * useMultiEvaluator.ts
 *
 * React hook for interacting with the multi-evaluator API
 */

import { useState } from 'react';

interface MultiEvalRequest {
  input_data: string;
  output: string;
  use_legacy_format?: boolean;
}

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

interface ComparisonResult {
  multi_evaluator: MultiEvalResult;
  monolithic?: any;
  comparison: {
    cost_savings_pct?: number;
    latency_improvement_pct?: number;
    verdict_agreement?: boolean;
    multi_cost?: number;
    monolithic_cost?: number;
  };
}

const API_BASE = '/api/projects';

export const useMultiEvaluator = (projectId: string) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Evaluate using multi-evaluator system
   */
  const evaluate = async (
    inputData: string,
    output: string,
    useLegacyFormat = false
  ): Promise<MultiEvalResult | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/${projectId}/multi-eval/evaluate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_data: inputData,
            output: output,
            use_legacy_format: useLegacyFormat
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Evaluation failed');
      }

      const result = await response.json();
      setLoading(false);
      return result;
    } catch (err: any) {
      setError(err.message || 'Unknown error');
      setLoading(false);
      return null;
    }
  };

  /**
   * Preview which dimensions will be evaluated
   */
  const detectDimensions = async (): Promise<AutoDetectResult | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/${projectId}/multi-eval/detect-dimensions`
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Detection failed');
      }

      const result = await response.json();
      setLoading(false);
      return result;
    } catch (err: any) {
      setError(err.message || 'Unknown error');
      setLoading(false);
      return null;
    }
  };

  /**
   * Compare multi-evaluator vs monolithic
   */
  const compare = async (
    inputData: string,
    output: string
  ): Promise<ComparisonResult | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/${projectId}/multi-eval/compare`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_data: inputData,
            output: output
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Comparison failed');
      }

      const result = await response.json();
      setLoading(false);
      return result;
    } catch (err: any) {
      setError(err.message || 'Unknown error');
      setLoading(false);
      return null;
    }
  };

  /**
   * Batch evaluate multiple inputs
   */
  const batchEvaluate = async (
    requests: MultiEvalRequest[]
  ): Promise<MultiEvalResult[] | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/${projectId}/multi-eval/batch`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requests)
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Batch evaluation failed');
      }

      const results = await response.json();
      setLoading(false);
      return results;
    } catch (err: any) {
      setError(err.message || 'Unknown error');
      setLoading(false);
      return null;
    }
  };

  return {
    evaluate,
    detectDimensions,
    compare,
    batchEvaluate,
    loading,
    error
  };
};

export default useMultiEvaluator;
