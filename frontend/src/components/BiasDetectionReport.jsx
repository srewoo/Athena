import React from 'react';
import { 
  AlertTriangle, 
  CheckCircle, 
  AlertCircle, 
  TrendingUp, 
  TrendingDown,
  Info,
  X
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
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';

/**
 * BiasDetectionReport Component
 * 
 * Displays bias detection results for eval prompts.
 * Shows detected biases, severity, evidence, and recommendations.
 */
export const BiasDetectionReport = ({ report, onClose, isOpen = false }) => {
  if (!report) return null;

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'secondary';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical':
      case 'high':
        return <AlertTriangle className="h-4 w-4" />;
      case 'medium':
        return <AlertCircle className="h-4 w-4" />;
      case 'low':
        return <Info className="h-4 w-4" />;
      default:
        return <CheckCircle className="h-4 w-4" />;
    }
  };

  const getBiasTypeLabel = (biasType) => {
    const labels = {
      'position_bias': 'Position Bias',
      'length_bias': 'Length Bias',
      'verbosity_bias': 'Verbosity Bias',
      'confirmation_bias': 'Confirmation Bias',
      'leniency_bias': 'Leniency Bias',
      'severity_bias': 'Severity Bias',
      'anchoring_bias': 'Anchoring Bias'
    };
    return labels[biasType] || biasType;
  };

  const detectedBiases = report.biases_detected?.filter(b => b.detected) || [];
  const totalBiases = report.biases_detected?.length || 0;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {report.is_biased ? (
              <AlertTriangle className="h-5 w-5 text-orange-600" />
            ) : (
              <CheckCircle className="h-5 w-5 text-green-600" />
            )}
            Bias Detection Report
          </DialogTitle>
          <DialogDescription>
            Comprehensive bias analysis for your evaluation prompt
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Overall Score */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Overall Bias Score</CardTitle>
              <CardDescription>
                0 = No bias, 100 = Severe bias
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-3xl font-bold">
                    {report.overall_bias_score}
                    <span className="text-base text-muted-foreground ml-1">/ 100</span>
                  </span>
                  <Badge 
                    variant={report.overall_bias_score > 50 ? 'destructive' : 
                            report.overall_bias_score > 30 ? 'default' : 'secondary'}
                    className="text-lg px-3 py-1"
                  >
                    {report.is_biased ? 'Biased' : 'Acceptable'}
                  </Badge>
                </div>
                <Progress 
                  value={report.overall_bias_score} 
                  className={`h-3 ${
                    report.overall_bias_score > 50 ? 'bg-red-200' :
                    report.overall_bias_score > 30 ? 'bg-yellow-200' :
                    'bg-green-200'
                  }`}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>No bias</span>
                  <span>Moderate</span>
                  <span>Severe</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Critical Biases Alert */}
          {report.critical_biases && report.critical_biases.length > 0 && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <strong>Critical biases detected:</strong>{' '}
                {report.critical_biases.map(getBiasTypeLabel).join(', ')}
                <br />
                <span className="text-sm">
                  These should be addressed before deploying this eval prompt.
                </span>
              </AlertDescription>
            </Alert>
          )}

          {/* Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Detection Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold">{totalBiases}</div>
                  <div className="text-sm text-muted-foreground">Tests Run</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-orange-600">
                    {detectedBiases.length}
                  </div>
                  <div className="text-sm text-muted-foreground">Biases Found</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-red-600">
                    {report.critical_biases?.length || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">Critical</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Detected Biases */}
          {detectedBiases.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Detected Biases</CardTitle>
                <CardDescription>
                  Biases that were found in your evaluation prompt
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {detectedBiases.map((bias, index) => (
                  <div key={index} className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant={getSeverityColor(bias.severity)}>
                          {getSeverityIcon(bias.severity)}
                          <span className="ml-1">{bias.severity}</span>
                        </Badge>
                        <h4 className="font-semibold">
                          {getBiasTypeLabel(bias.bias_type)}
                        </h4>
                      </div>
                      <Badge variant="outline">
                        {(bias.confidence * 100).toFixed(0)}% confidence
                      </Badge>
                    </div>

                    {/* Evidence */}
                    <div className="text-sm space-y-1">
                      <div className="font-medium text-muted-foreground">Evidence:</div>
                      <ul className="list-disc list-inside space-y-0.5">
                        {bias.evidence?.map((ev, i) => (
                          <li key={i} className="text-foreground">{ev}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Metrics */}
                    {bias.metrics && Object.keys(bias.metrics).length > 0 && (
                      <div className="text-sm">
                        <div className="font-medium text-muted-foreground">Metrics:</div>
                        <div className="flex gap-4 mt-1">
                          {Object.entries(bias.metrics).slice(0, 3).map(([key, value]) => (
                            <div key={key} className="text-xs">
                              <span className="text-muted-foreground">{key}:</span>{' '}
                              <span className="font-medium">
                                {typeof value === 'number' ? value.toFixed(3) : JSON.stringify(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Recommendation */}
                    {bias.recommendation && (
                      <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription className="text-sm">
                          <strong>Recommendation:</strong> {bias.recommendation.split('\n')[0]}
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* No Biases */}
          {detectedBiases.length === 0 && (
            <Alert>
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription>
                No significant biases detected! Your evaluation prompt appears to be unbiased.
              </AlertDescription>
            </Alert>
          )}

          {/* Recommendations */}
          {report.recommendations && report.recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {report.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <div className="mt-0.5">
                        {rec.includes('CRITICAL') || rec.includes('⚠️') ? (
                          <AlertTriangle className="h-4 w-4 text-orange-600" />
                        ) : (
                          <CheckCircle className="h-4 w-4 text-blue-600" />
                        )}
                      </div>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Test Metadata */}
          {report.test_metadata && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-normal text-muted-foreground">
                  Test Metadata
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {Object.entries(report.test_metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-muted-foreground">{key}:</span>
                      <span className="font-medium">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <DialogFooter>
          <Button onClick={onClose} variant="outline">
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default BiasDetectionReport;
