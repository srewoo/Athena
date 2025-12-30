/**
 * API Utility for Athena Frontend
 * Handles all API calls with error handling and retry logic
 */

import { API } from "../App";
import { ERROR_MESSAGES, STORAGE_KEYS } from "./constants";

// Request timeout (30 seconds for normal, 5 minutes for LLM calls)
const DEFAULT_TIMEOUT = 30000;
const LLM_TIMEOUT = 300000;

/**
 * Make an API request with proper error handling
 */
export async function apiRequest(
  endpoint,
  options = {},
  {
    timeout = DEFAULT_TIMEOUT,
    retries = 2,
    retryDelay = 1000
  } = {}
) {
  const url = endpoint.startsWith("http") ? endpoint : `${API}${endpoint}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  const defaultOptions = {
    headers: {
      "Content-Type": "application/json",
    },
    signal: controller.signal,
  };

  const mergedOptions = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  };

  let lastError;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, mergedOptions);
      clearTimeout(timeoutId);

      // Handle rate limiting
      if (response.status === 429) {
        const resetTime = response.headers.get("X-RateLimit-Reset");
        const error = new Error(ERROR_MESSAGES.RATE_LIMIT);
        error.status = 429;
        error.resetTime = resetTime;
        throw error;
      }

      // Handle auth errors
      if (response.status === 401) {
        const error = new Error(ERROR_MESSAGES.AUTH_ERROR);
        error.status = 401;
        throw error;
      }

      // Handle validation errors
      if (response.status === 400) {
        const data = await response.json();
        const error = new Error(data.detail || ERROR_MESSAGES.VALIDATION_ERROR);
        error.status = 400;
        error.details = data;
        throw error;
      }

      // Handle server errors with retry
      if (response.status >= 500) {
        if (attempt < retries) {
          await sleep(retryDelay * (attempt + 1));
          continue;
        }
        const error = new Error(ERROR_MESSAGES.SERVER_ERROR);
        error.status = response.status;
        throw error;
      }

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        const error = new Error(data.detail || `Request failed: ${response.status}`);
        error.status = response.status;
        throw error;
      }

      return await response.json();

    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === "AbortError") {
        throw new Error("Request timed out. Please try again.");
      }

      if (error.message === "Failed to fetch") {
        if (attempt < retries) {
          await sleep(retryDelay * (attempt + 1));
          continue;
        }
        throw new Error(ERROR_MESSAGES.NETWORK_ERROR);
      }

      lastError = error;

      // Don't retry for client errors
      if (error.status && error.status < 500) {
        throw error;
      }

      // Retry for other errors
      if (attempt < retries) {
        await sleep(retryDelay * (attempt + 1));
        continue;
      }
    }
  }

  throw lastError || new Error("Request failed after retries");
}

/**
 * GET request
 */
export async function get(endpoint, options = {}) {
  return apiRequest(endpoint, { method: "GET", ...options });
}

/**
 * POST request
 */
export async function post(endpoint, data, options = {}) {
  return apiRequest(endpoint, {
    method: "POST",
    body: JSON.stringify(data),
    ...options,
  });
}

/**
 * PUT request
 */
export async function put(endpoint, data, options = {}) {
  return apiRequest(endpoint, {
    method: "PUT",
    body: JSON.stringify(data),
    ...options,
  });
}

/**
 * DELETE request
 */
export async function del(endpoint, options = {}) {
  return apiRequest(endpoint, { method: "DELETE", ...options });
}

/**
 * POST request for LLM operations (longer timeout)
 */
export async function postLLM(endpoint, data, options = {}) {
  return apiRequest(
    endpoint,
    {
      method: "POST",
      body: JSON.stringify(data),
      ...options,
    },
    { timeout: LLM_TIMEOUT, retries: 1 }
  );
}

/**
 * Sleep utility
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Get stored settings
 */
export function getStoredSettings() {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.LLM_SETTINGS);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (e) {
    console.error("Failed to parse stored settings:", e);
  }
  return null;
}

/**
 * Save settings to localStorage and sync with backend
 */
export async function saveSettings(settings) {
  // Save to localStorage
  localStorage.setItem(STORAGE_KEYS.LLM_SETTINGS, JSON.stringify(settings));

  // Sync to backend
  try {
    await post("/settings", settings);
  } catch (e) {
    console.error("Failed to sync settings to backend:", e);
    // Don't throw - localStorage save is the primary storage
  }

  return settings;
}

/**
 * Load settings from backend and cache in localStorage
 */
export async function loadSettings() {
  // Try localStorage first
  const stored = getStoredSettings();
  if (stored && stored.api_key) {
    // Sync to backend in background
    post("/settings", stored).catch(() => {});
    return stored;
  }

  // Fall back to backend
  try {
    const settings = await get("/settings");
    if (settings && settings.api_key) {
      localStorage.setItem(STORAGE_KEYS.LLM_SETTINGS, JSON.stringify(settings));
      return settings;
    }
  } catch (e) {
    console.error("Failed to load settings from backend:", e);
  }

  return null;
}

/**
 * Analyze a prompt with optional enhanced (LLM-powered) analysis
 * @param {string} promptText - The prompt to analyze
 * @param {object} options - Optional parameters
 * @param {boolean} options.enhanced - Use LLM-enhanced analysis (default: true)
 * @param {string} options.useCase - Context about the prompt's use case
 * @param {string} options.requirements - Requirements the prompt should meet
 */
export async function analyzePrompt(promptText, options = {}) {
  const { enhanced = true, useCase = "", requirements = "" } = options;

  return postLLM("/api/step2/analyze", {
    prompt_text: promptText,
    use_case: useCase,
    requirements: requirements,
    enhanced: enhanced
  });
}

/**
 * Deep analyze a prompt using hybrid Python+LLM analysis
 * Always uses enhanced analysis (requires API key configured)
 */
export async function analyzePromptEnhanced(promptText, useCase = "", requirements = "") {
  return postLLM("/api/step2/analyze-enhanced", {
    prompt_text: promptText,
    use_case: useCase,
    requirements: requirements
  });
}

// Export API base URL for components that need it
export { API };
