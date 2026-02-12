import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { toast } from 'sonner';
import DatasetGeneration from '../DatasetGeneration';

jest.mock('axios');

describe('DatasetGeneration Component', () => {
  const mockProject = {
    id: 'project-123',
    name: 'Test Project',
  };

  const mockSelectedVersion = {
    id: 'version-123',
    content: 'Test system prompt',
  };

  const mockSettings = {
    openai_key: 'test-openai-key',
    claude_key: 'test-claude-key',
    gemini_key: 'test-gemini-key',
    default_provider: 'openai',
    default_model: 'gpt-4o-mini',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    axios.get.mockResolvedValue({ data: [] });
  });

  test('renders dataset generation interface', () => {
    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
      />
    );

    expect(screen.getByText('Step 4: Test Dataset Generation')).toBeInTheDocument();
    expect(screen.getByLabelText(/Sample Count/i)).toBeInTheDocument();
  });

  test('shows error when project is not provided', () => {
    render(<DatasetGeneration project={null} selectedVersion={null} settings={mockSettings} />);

    expect(screen.getByText(/Please create a project first/i)).toBeInTheDocument();
  });

  test('validates sample count is within range', async () => {
    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
      />
    );

    const sampleInput = screen.getByLabelText(/Sample Count/i);
    fireEvent.change(sampleInput, { target: { value: '150' } });

    const generateButton = screen.getByRole('button', { name: /Generate Test Cases/i });
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('between 1 and 100'));
    });
  });

  test('uses correct API key based on provider', async () => {
    const anthropicSettings = {
      ...mockSettings,
      default_provider: 'anthropic',
    };

    axios.post.mockResolvedValueOnce({ data: [] });

    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={anthropicSettings}
      />
    );

    const generateButton = screen.getByRole('button', { name: /Generate Test Cases/i });
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/test-cases'),
        expect.objectContaining({
          provider: 'anthropic',
          api_key: 'test-claude-key',
        })
      );
    });
  });

  test('shows specific error for missing API key', async () => {
    const noKeySettings = {
      ...mockSettings,
      default_provider: 'google',
      gemini_key: null,
    };

    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={noKeySettings}
      />
    );

    const generateButton = screen.getByRole('button', { name: /Generate Test Cases/i });
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('GOOGLE API key'));
    });
  });

  test('successfully generates test cases', async () => {
    const mockTestCases = [
      {
        id: 'test-1',
        input_text: 'Test input 1',
        expected_behavior: 'Expected behavior 1',
        case_type: 'positive',
      },
      {
        id: 'test-2',
        input_text: 'Test input 2',
        expected_behavior: 'Expected behavior 2',
        case_type: 'edge',
      },
    ];

    axios.post.mockResolvedValueOnce({ data: mockTestCases });

    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
      />
    );

    const generateButton = screen.getByRole('button', { name: /Generate Test Cases/i });
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('Generated 2 test cases'));
    });
  });

  test('handles generation errors gracefully', async () => {
    axios.post.mockRejectedValueOnce({
      response: { data: { detail: 'API error' } },
    });

    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
      />
    );

    const generateButton = screen.getByRole('button', { name: /Generate Test Cases/i });
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('API error'));
    });
  });

  test('clamps sample count to valid range', async () => {
    render(
      <DatasetGeneration
        project={mockProject}
        selectedVersion={mockSelectedVersion}
        settings={mockSettings}
      />
    );

    const sampleInput = screen.getByLabelText(/Sample Count/i);
    
    // Test upper bound
    fireEvent.change(sampleInput, { target: { value: '200' } });
    expect(sampleInput.value).toBe('100');

    // Test lower bound
    fireEvent.change(sampleInput, { target: { value: '-5' } });
    expect(sampleInput.value).toBe('1');
  });
});
