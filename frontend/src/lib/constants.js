/**
 * Shared Constants for Athena Frontend
 */

// Model options per provider (2025 releases)
export const MODEL_OPTIONS = {
  openai: [
    { value: "gpt-4o", label: "GPT-4o (Recommended)", description: "Most capable GPT-4 model" },
    { value: "gpt-4o-mini", label: "GPT-4o Mini", description: "Faster, cost-effective" },
    { value: "gpt-4-turbo", label: "GPT-4 Turbo", description: "Previous generation turbo" },
    { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo", description: "Fast and cheap" },
    { value: "__divider_reasoning__", label: "--- Reasoning Models ---", disabled: true },
    { value: "o1", label: "O1", description: "Advanced reasoning" },
    { value: "o1-mini", label: "O1 Mini", description: "Fast reasoning" },
    { value: "o1-preview", label: "O1 Preview", description: "Reasoning preview" },
    { value: "o3-mini", label: "O3 Mini", description: "Next-gen reasoning" },
  ],
  claude: [
    { value: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet (Recommended)", description: "Best for coding" },
    { value: "claude-3-opus-20240229", label: "Claude 3 Opus", description: "Most capable Claude" },
    { value: "claude-3-sonnet-20240229", label: "Claude 3 Sonnet", description: "Balanced" },
    { value: "claude-3-haiku-20240307", label: "Claude 3 Haiku", description: "Fast and cheap" },
  ],
  gemini: [
    { value: "gemini-2.0-flash-exp", label: "Gemini 2.0 Flash (Recommended)", description: "Latest flash model" },
    { value: "gemini-1.5-pro", label: "Gemini 1.5 Pro", description: "Most capable" },
    { value: "gemini-1.5-flash", label: "Gemini 1.5 Flash", description: "Fast and efficient" },
    { value: "gemini-1.0-pro", label: "Gemini 1.0 Pro", description: "Previous generation" },
  ]
};

// Thinking/Reasoning models for evaluation
export const THINKING_MODELS = {
  openai: ["o1", "o1-mini", "o1-preview", "o3-mini"],
  claude: ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
  gemini: ["gemini-1.5-pro"]
};

// Provider display names
export const PROVIDER_NAMES = {
  openai: "OpenAI",
  claude: "Anthropic Claude",
  gemini: "Google Gemini"
};

// Default models per provider
export const DEFAULT_MODELS = {
  openai: "gpt-4o",
  claude: "claude-3-5-sonnet-20241022",
  gemini: "gemini-2.0-flash-exp"
};

// Test case categories
export const TEST_CATEGORIES = {
  positive: { label: "Positive", color: "green", description: "Standard valid inputs" },
  edge_case: { label: "Edge Case", color: "yellow", description: "Boundary conditions" },
  negative: { label: "Negative", color: "red", description: "Invalid inputs" },
  adversarial: { label: "Adversarial", color: "purple", description: "Security/robustness tests" }
};

// Evaluation verdicts
export const VERDICTS = {
  PASS: { label: "Pass", color: "green", bgColor: "bg-green-100", textColor: "text-green-800" },
  NEEDS_REVIEW: { label: "Needs Review", color: "yellow", bgColor: "bg-yellow-100", textColor: "text-yellow-800" },
  FAIL: { label: "Fail", color: "red", bgColor: "bg-red-100", textColor: "text-red-800" }
};

// Quality score thresholds
export const SCORE_THRESHOLDS = {
  excellent: 4.5,
  good: 4.0,
  acceptable: 3.0,
  poor: 2.0
};

// API error messages
export const ERROR_MESSAGES = {
  RATE_LIMIT: "Too many requests. Please wait a moment and try again.",
  AUTH_ERROR: "Invalid API key. Please check your settings.",
  NETWORK_ERROR: "Network error. Please check your connection.",
  SERVER_ERROR: "Server error. Please try again later.",
  VALIDATION_ERROR: "Invalid input. Please check your data."
};

// Storage keys
export const STORAGE_KEYS = {
  LLM_SETTINGS: "athena_llm_settings",
  THEME: "athena-theme",
  RECENT_PROJECTS: "athena_recent_projects"
};

// API endpoints
export const ENDPOINTS = {
  SETTINGS: "/settings",
  PROJECTS: "/projects",
  ANALYZE: "/step2/analyze",
  ANALYZE_ENHANCED: "/step2/analyze-enhanced",
  OPTIMIZE: "/step2/optimize",
  AGENTIC_REWRITE: "/step2/agentic-rewrite",
  GENERATE_EVAL: "/step3/agentic-generate-eval",
  GENERATE_EVAL_V2: "/step3/generate-eval-v2",
  GENERATE_DATASET: "/step4/smart-generate-testdata",
  EXECUTE_TESTS: "/step5/execute-tests",
  AB_TEST: "/ab-test",
  PLAYGROUND: "/playground"
};

// Helper function to get model label
export function getModelLabel(provider, modelValue) {
  const models = MODEL_OPTIONS[provider] || [];
  const model = models.find(m => m.value === modelValue);
  return model ? model.label : modelValue;
}

// Helper function to check if model is a reasoning model
export function isReasoningModel(provider, modelValue) {
  const thinkingModels = THINKING_MODELS[provider] || [];
  return thinkingModels.some(m => modelValue?.startsWith(m));
}

// Helper function to get default model for provider
export function getDefaultModel(provider) {
  return DEFAULT_MODELS[provider] || MODEL_OPTIONS[provider]?.[0]?.value || "";
}
