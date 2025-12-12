# Athena

## Your Strategic Prompt Architect

A comprehensive AI-powered platform for prompt engineering, optimization, testing, and evaluation. Athena provides a complete workflow for developing, refining, and validating system prompts with iterative feedback loops and AI-powered analysis.

## Features

### Core Workflow (5-Step Process)

**Step 1: Project Setup**
- Create projects with name, use case, and requirements
- Define initial system prompts
- Auto-save projects to JSON storage

**Step 2: Prompt Optimization**
- AI-powered prompt analysis using heuristics + LLM
- Hybrid scoring (word count, best practices, AI insights)
- Actionable suggestions with priority levels (High/Medium/Low)
- LLM-enhanced insights: strengths, issues, analysis summary
- Manual feedback refinement with iteration tracking
- AI Auto-Rewrite for autonomous improvement

**Step 3: Evaluation Prompt Generation**
- Generate evaluation prompts with 5-section structure:
  - Role & Goal definition
  - Core Expectations
  - Detailed 1-5 Rating Scale with failure modes
  - Evaluation Task steps
  - Output Format (JSON)
- Iterative refinement with user feedback

**Step 4: Test Dataset Generation**
- AI-generated test cases based on selected prompt version
- Distribution-based generation:
  - 60% positive/typical cases
  - 20% edge cases/boundary conditions
  - 10% negative/inappropriate inputs
  - 10% adversarial/injection attempts
- Customizable sample count

**Step 5: Test Execution & Results**
- Execute tests against system prompt
- Evaluate outputs using generated eval prompt
- Interactive results table with expandable rows
- Summary statistics and pass rates
- Export results as JSON

### Additional Features

- **Version Management**: Track prompt iterations with full history
- **Dark/Light Theme**: Toggle between themes
- **Settings Persistence**: API keys saved in browser localStorage, auto-synced to backend
- **Project Management**: Save, load, and manage multiple projects
- **Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini

### Analysis Tools

- **Prompt Analysis**: Hybrid heuristic + LLM analysis
- **Contradiction Detection**: Identify conflicting instructions
- **Delimiter Analysis**: Analyze XML/Markdown structure usage
- **Metaprompt Generation**: AI-generated improvement suggestions

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | Python web framework |
| Pydantic | Data validation |
| OpenAI SDK | GPT-4/GPT-4o integration |
| Anthropic SDK | Claude integration |
| Google Generative AI | Gemini integration |
| File-based Storage | JSON project storage |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| shadcn/ui | Component library |
| Tailwind CSS | Utility-first styling |
| Radix UI | Accessible primitives |
| Lucide React | Icons |

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm/yarn**
- **API Key** for at least one LLM provider:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Anthropic API Key](https://console.anthropic.com/)
  - [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

## Installation

### Step 1: Clone/Navigate to Repository

```bash
cd /path/to/Athena
```

### Step 2: Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install
# OR
yarn install
```

## Running the Application

### Start Backend Server

```bash
cd backend
source venv/bin/activate
python server.py
```

Backend available at: `http://localhost:8000`

### Start Frontend Server

In a separate terminal:

```bash
cd frontend
npm start
# OR
yarn start
```

Frontend available at: `http://localhost:3000`

## Quick Start Guide

### 1. Configure API Settings

1. Open `http://localhost:3000`
2. Click Settings icon (gear) in top-right
3. Select your LLM provider (OpenAI/Claude/Gemini)
4. Enter your API key
5. Click "Save Settings"

Settings persist in browser localStorage and auto-sync to backend on page load.

### 2. Create a Project

1. Enter project name
2. Define use case (e.g., "Customer support chatbot")
3. Add key requirements
4. Write initial system prompt
5. Click "Next: Optimize Prompt"

### 3. Optimize Your Prompt

1. Click "Re-Analyze" to get AI-powered analysis
2. Review scores and suggestions
3. Use "Review & Refine" for manual feedback
4. Use "AI Rewrite" for autonomous improvement
5. Click "Continue to Eval Prompt" when satisfied

### 4. Generate Evaluation Criteria

1. Click "Generate Evaluation Prompt"
2. Review the 5-section eval structure
3. Provide feedback to refine criteria
4. Proceed to test dataset generation

### 5. Generate Test Dataset

1. Set number of samples (default: 10)
2. Click "Generate Dataset"
3. Review generated test cases
4. Test cases are generated based on selected prompt version

### 6. Execute Tests

1. Run tests against your system prompt
2. Review results with pass/fail status
3. Export results as needed

## Project Structure

```
Athena/
├── .env.example               # Environment template (copy to .env)
├── .env                       # Your configuration (create from example)
├── README.md
├── backend/
│   ├── server.py              # Main FastAPI server (1300+ lines)
│   ├── project_api.py         # Project management API (900+ lines)
│   ├── project_storage.py     # JSON file storage
│   ├── llm_client.py          # Multi-provider LLM client
│   ├── models.py              # Pydantic models
│   ├── shared_settings.py     # Settings persistence module
│   ├── requirements.txt       # Python dependencies
│   └── saved_projects/        # Project JSON files
└── frontend/
    ├── src/
    │   ├── App.js             # Main application (exports API, BASE_URL)
    │   ├── pages/
    │   │   ├── PromptOptimizer.js  # Main workflow UI
    │   │   ├── Dashboard.js        # Dashboard view
    │   │   ├── Playground.js       # Prompt testing
    │   │   └── ...
    │   └── components/        # UI components
    ├── package.json           # Uses dotenv-cli to load ../.env
    └── tailwind.config.js
```

## API Endpoints

### Settings
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings` | Get LLM configuration |
| POST | `/api/settings` | Save LLM configuration |

### Projects
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create new project |
| GET | `/api/projects/{id}` | Get project details |
| PUT | `/api/projects/{id}` | Update project |
| DELETE | `/api/projects/{id}` | Delete project |

### Analysis & Optimization
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/{id}/analyze` | Hybrid heuristic + LLM analysis |
| POST | `/api/projects/{id}/rewrite` | AI-powered prompt rewrite |
| POST | `/api/rewrite` | Global prompt rewrite |

### Evaluation & Testing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/{id}/eval-prompt/generate` | Generate eval prompt |
| POST | `/api/projects/{id}/eval-prompt/refine` | Refine eval with feedback |
| POST | `/api/projects/{id}/dataset/generate` | Generate test dataset |
| POST | `/api/projects/{id}/test-runs` | Create test run |
| GET | `/api/projects/{id}/test-runs` | List test runs |

### Tools
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect-contradictions` | Detect conflicting instructions |
| POST | `/api/analyze-delimiters` | Analyze delimiter usage |
| POST | `/api/generate-metaprompt` | Generate improvement suggestions |
| POST | `/api/playground` | Test prompt with input |
| POST | `/api/evaluate` | Evaluate prompt quality |

### Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/export/json/{id}` | Export as JSON |
| GET | `/api/export/pdf/{id}` | Export as PDF |

## Environment Variables

All configuration is in a single `.env` file at the project root. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_HOST` | Backend server host | `0.0.0.0` |
| `BACKEND_PORT` | Backend server port | `8000` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:3000` |
| `REACT_APP_BASE_URL` | Backend URL for frontend | `http://localhost:8000` |

### Optional LLM Configuration

You can also set LLM credentials in `.env` (or configure via UI):

```bash
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=AI...
```

## Default Models

| Provider | Default Model |
|----------|---------------|
| OpenAI | `gpt-4o-mini` |
| Anthropic | `claude-3-sonnet-20240229` |
| Google | `gemini-1.5-flash` |

Custom models can be specified in Settings.

## Troubleshooting

### Backend Won't Start
```bash
# Check if port 8000 is in use
lsof -ti:8000

# Kill existing process
lsof -ti:8000 | xargs kill -9

# Restart backend
cd backend && source venv/bin/activate && python server.py
```

### API Key Issues
- Settings persist in browser localStorage
- After server restart, refresh the page to auto-sync settings
- Check Settings modal to verify API key is saved

### Frontend Issues
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules
npm install
npm start
```

### LLM Calls Not Working
1. Verify API key in Settings
2. Check browser console for errors
3. Ensure backend shows "Settings synced from localStorage"

## Development

### Backend Development
```bash
cd backend
source venv/bin/activate
python server.py
# Server auto-reloads on file changes
```

### Frontend Development
```bash
cd frontend
npm start
# React hot-reloads on file changes
```

### Running Tests
```bash
cd backend
pytest test_endpoints.py -v
```

## Architecture Highlights

### Settings Persistence
- API keys saved to browser localStorage (`athena_llm_settings`)
- Auto-synced to backend on page load
- Survives browser refresh, close, and server restart

### Hybrid Analysis
- Step 1: Fast heuristic analysis (word count, keyword checks)
- Step 2: LLM-enhanced insights (strengths, issues, refined score)
- Step 3: Combined suggestions with deduplication

### Version Management
- All prompt versions stored in `system_prompt_versions` array
- Each version tracked with timestamp and version number
- Test datasets generated against specific versions

## License

This project is for educational and development purposes.
