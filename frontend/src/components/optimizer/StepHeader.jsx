/**
 * Step Header Component
 * Collapsible section header for each workflow step
 */
import React from "react";
import { ChevronDown, ChevronRight, CheckCircle, AlertCircle, RotateCcw } from "lucide-react";
import { Button } from "../ui/button";

const StepHeader = ({
  stepNumber,
  title,
  description,
  isExpanded,
  onToggle,
  isComplete,
  hasError,
  onRegenerate,
  isRegenerating,
  children
}) => {
  const getStatusIcon = () => {
    if (hasError) {
      return <AlertCircle className="h-5 w-5 text-destructive" />;
    }
    if (isComplete) {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    }
    return null;
  };

  return (
    <div className="border rounded-lg">
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-accent/50 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary font-semibold">
            {stepNumber}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-medium">{title}</h3>
              {getStatusIcon()}
            </div>
            {description && (
              <p className="text-sm text-muted-foreground">{description}</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isComplete && onRegenerate && (
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onRegenerate();
              }}
              disabled={isRegenerating}
              title="Regenerate"
            >
              <RotateCcw className={`h-4 w-4 ${isRegenerating ? 'animate-spin' : ''}`} />
            </Button>
          )}
          {isExpanded ? (
            <ChevronDown className="h-5 w-5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-5 w-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 pt-0 border-t">
          {children}
        </div>
      )}
    </div>
  );
};

export default StepHeader;
