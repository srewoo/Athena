import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { toast } from 'sonner';
import PromptOptimization from '../PromptOptimization';

jest.mock('axios');

describe('PromptOptimization Component', () => {
  const mockProject = {
    id: 'project-123',
    name: 'Test Project',
  };

  const mockSelectedVersion = {
    id: 'version-123',
    content: 'Test system prompt for optimization',
  };

  const mockSettings = {
    openai_key: 'test-openai-key',
    claude_key: 'test-claude-key',
    gemini_key: 'test-gemini-key',
    default_provider: 'openai',
    default_model: 'gpt-4o-mini',
  };

  const mockOnVersionCreated = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    axios.get.mockResolvedValue({ data: [] });
  });

  test('renders prompt optimization interface', () => {
    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    expect(screen.getByText('Step 2: Prompt Optimization')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Analyze/i })).toBeInTheDocument();
  });

  test('shows error when prompt is empty', async () => {
    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={{ ...mockSelectedVersion, content: '' }}
        settings={mockSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('enter a prompt'));
    });
  });

  test('uses correct API key for OpenAI provider', async () => {
    axios.post.mockResolvedValueOnce({
      data: {
        combined_score: 8.5,
        suggestions: [],
        issues: [],
        strengths: [],
      },
    });

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/analyze'),
        expect.objectContaining({
          provider: 'openai',
          api_key: 'test-openai-key',
        })
      );
    });
  });

  test('uses correct API key for Anthropic provider', async () => {
    const anthropicSettings = {
      ...mockSettings,
      default_provider: 'anthropic',
    };

    axios.post.mockResolvedValueOnce({
      data: {
        combined_score: 9.0,
        suggestions: [],
        issues: [],
        strengths: [],
      },
    });

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={anthropicSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/analyze'),
        expect.objectContaining({
          provider: 'anthropic',
          api_key: 'test-claude-key',
        })
      );
    });
  });

  test('uses correct API key for Google provider', async () => {
    const googleSettings = {
      ...mockSettings,
      default_provider: 'google',
    };

    axios.post.mockResolvedValueOnce({
      data: {
        combined_score: 8.7,
        suggestions: [],
        issues: [],
        strengths: [],
      },
    });

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={googleSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/analyze'),
        expect.objectContaining({
          provider: 'google',
          api_key: 'test-gemini-key',
        })
      );
    });
  });

  test('shows provider-specific error when API key is missing', async () => {
    const noKeySettings = {
      ...mockSettings,
      default_provider: 'anthropic',
      claude_key: null,
    };

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={noKeySettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('ANTHROPIC API key'));
    });
  });

  test('successfully analyzes prompt and displays results', async () => {
    const mockAnalysis = {
      combined_score: 7.5,
      suggestions: ['Suggestion 1', 'Suggestion 2'],
      issues: ['Issue 1'],
      strengths: ['Strength 1', 'Strength 2'],
    };

    axios.post.mockResolvedValueOnce({ data: mockAnalysis });

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(toast.success).toHaveBeenCalled();
      expect(screen.getByText(/7.5/)).toBeInTheDocument();
    });
  });

  test('auto-rewrite uses correct API key for rewrite operation', async () => {
    const mockAnalysis = {
      combined_score: 7.5,
      suggestions: ['Improve clarity'],
      issues: ['Too verbose'],
      strengths: ['Good structure'],
    };

    axios.post
      .mockResolvedValueOnce({ data: mockAnalysis }) // analyze call
      .mockResolvedValueOnce({
        // rewrite call
        data: { rewritten_prompt: 'Improved prompt' },
      })
      .mockResolvedValueOnce({
        // create version call
        data: { id: 'new-version', content: 'Improved prompt' },
      });

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    // First analyze
    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(screen.getByText(/7.5/)).toBeInTheDocument();
    });

    // Then auto-rewrite
    const rewriteButton = screen.getByRole('button', { name: /Auto-Rewrite/i });
    fireEvent.click(rewriteButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/rewrite'),
        expect.objectContaining({
          provider: 'openai',
          api_key: 'test-openai-key',
        })
      );
    });
  });

  test('handles analysis errors gracefully', async () => {
    axios.post.mockRejectedValueOnce({
      response: { data: { detail: 'Analysis failed' } },
    });

    render(
      <PromptOptimization
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
        onVersionCreated={mockOnVersionCreated}
      />
    );

    const analyzeButton = screen.getByRole('button', { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('Analysis failed'));
    });
  });
});
