import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { toast } from 'sonner';
import ProjectSetup from '../ProjectSetup';

jest.mock('axios');

describe('ProjectSetup Component', () => {
  const mockOnProjectCreated = jest.fn();
  const mockSessionId = 'test-session-123';
  const mockSettings = {
    openai_key: 'test-openai-key',
    claude_key: 'test-claude-key',
    gemini_key: 'test-gemini-key',
    default_provider: 'openai',
    default_model: 'gpt-4o-mini',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders project setup form', () => {
    render(
      <ProjectSetup
        onProjectCreated={mockOnProjectCreated}
        sessionId={mockSessionId}
        settings={mockSettings}
      />
    );

    expect(screen.getByText('Step 1: Project Setup')).toBeInTheDocument();
    expect(screen.getByLabelText(/Project Name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Initial System Prompt/i)).toBeInTheDocument();
  });

  test('validates required fields before submission', async () => {
    render(
      <ProjectSetup
        onProjectCreated={mockOnProjectCreated}
        sessionId={mockSessionId}
        settings={mockSettings}
      />
    );

    const continueButton = screen.getByRole('button', { name: /Continue/i });
    fireEvent.click(continueButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalled();
    });

    expect(mockOnProjectCreated).not.toHaveBeenCalled();
  });

  test('creates project with valid data', async () => {
    axios.post.mockResolvedValueOnce({
      data: {
        id: 'project-123',
        name: 'Test Project',
        use_case: 'Testing',
        requirements: 'Test requirements',
      },
    });

    render(
      <ProjectSetup
        onProjectCreated={mockOnProjectCreated}
        sessionId={mockSessionId}
        settings={mockSettings}
      />
    );

    const nameInput = screen.getByLabelText(/Project Name/i);
    const useCaseInput = screen.getByLabelText(/Use Case/i);
    const requirementsInput = screen.getByLabelText(/Key Requirements/i);
    const continueButton = screen.getByRole('button', { name: /Continue/i });

    fireEvent.change(nameInput, { target: { value: 'Test Project' } });
    fireEvent.change(useCaseInput, { target: { value: 'Testing' } });
    fireEvent.change(requirementsInput, { target: { value: 'Test requirements' } });
    fireEvent.click(continueButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/projects'),
        expect.objectContaining({
          name: 'Test Project',
          use_case: 'Testing',
          requirements: 'Test requirements',
        })
      );
      expect(mockOnProjectCreated).toHaveBeenCalled();
    });
  });

  test('shows error when API keys are not configured for auto-extraction', async () => {
    const noSettingsProps = {
      onProjectCreated: mockOnProjectCreated,
      sessionId: mockSessionId,
      settings: null,
    };

    render(<ProjectSetup {...noSettingsProps} />);

    const promptInput = screen.getByLabelText(/Initial System Prompt/i);
    fireEvent.change(promptInput, { target: { value: 'Test prompt' } });

    // Wait for debounce and extraction attempt
    await waitFor(
      () => {
        expect(toast.error).toHaveBeenCalledWith(
          expect.stringContaining('API keys')
        );
      },
      { timeout: 3000 }
    );
  });

  test('uses correct API key based on provider for auto-extraction', async () => {
    const anthropicSettings = {
      ...mockSettings,
      default_provider: 'anthropic',
    };

    axios.get.mockResolvedValueOnce({ data: anthropicSettings });
    axios.post.mockResolvedValueOnce({
      data: {
        success: true,
        use_case: 'Extracted use case',
        requirements: 'Extracted requirements',
      },
    });

    render(
      <ProjectSetup
        onProjectCreated={mockOnProjectCreated}
        sessionId={mockSessionId}
        settings={anthropicSettings}
      />
    );

    const promptInput = screen.getByLabelText(/Initial System Prompt/i);
    fireEvent.change(promptInput, { target: { value: 'Test prompt for extraction' } });

    await waitFor(
      () => {
        expect(axios.post).toHaveBeenCalledWith(
          expect.stringContaining('/extract-project-info'),
          expect.objectContaining({
            api_key: 'test-claude-key',
            provider: 'anthropic',
          })
        );
      },
      { timeout: 3000 }
    );
  });
});
