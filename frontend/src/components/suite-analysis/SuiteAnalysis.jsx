import { useState } from 'react';
import axios from 'axios';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { toast } from 'sonner';
import { Loader2, Shield, Sparkles } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function SuiteAnalysis({ project, evalPrompts }) {
  const [loading, setLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);

  const runAnalysis = async () => {
    if (!project || !evalPrompts || evalPrompts.length < 2) {
      toast.error('Need at least 2 evaluation prompts to analyze');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(
        `${API}/analyze-overlaps?project_id=${project.id}&similarity_threshold=0.7`
      );

      setAnalysisResults(response.data);

      toast.success('Analysis complete', {
        description: `Found ${response.data.overlap_count} potential overlaps`
      });
    } catch (error) {
      console.error('Error:', error);
      toast.error('Analysis failed', {
        description: error.response?.data?.detail || 'Please try again'
      });
    } finally {
      setLoading(false);
    }
  };

  if (!evalPrompts || evalPrompts.length < 2) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Shield className="w-5 h-5 text-primary" />
            <span>Suite Analysis</span>
          </CardTitle>
          <CardDescription>
            Generate at least 2 evaluation prompts to see suite analysis
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-primary" />
              <span>Suite Analysis</span>
            </CardTitle>
            <CardDescription>
              Analyze your evaluation suite for overlaps and quality
            </CardDescription>
          </div>
          <Button
            onClick={runAnalysis}
            disabled={loading}
            variant="outline"
            size="sm"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Run Analysis
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="text-center py-8">
            <Loader2 className="w-8 h-8 mx-auto mb-3 text-primary animate-spin" />
            <p className="text-sm text-muted-foreground">Analyzing suite...</p>
          </div>
        ) : analysisResults ? (
          <div className="space-y-4">
            <div className="p-4 border rounded-lg">
              <h3 className="font-medium mb-2">Overlap Analysis</h3>
              <p className="text-2xl font-bold text-primary">
                {analysisResults.overlap_count} overlaps detected
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                {analysisResults.summary}
              </p>
            </div>

            {analysisResults.overlaps && analysisResults.overlaps.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium text-sm">Overlapping Dimensions:</h4>
                {analysisResults.overlaps.map((overlap, idx) => (
                  <div key={idx} className="p-3 border rounded bg-orange-50 dark:bg-orange-950/20">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm">
                        {overlap.dimension_1} ↔ {overlap.dimension_2}
                      </span>
                      <span className="text-xs bg-orange-600 text-white px-2 py-1 rounded">
                        {Math.round(overlap.similarity * 100)}% similar
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {overlap.recommendation}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <p className="text-sm">Click "Run Analysis" to check for overlaps and quality issues</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
