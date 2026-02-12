// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock environment variables
process.env.REACT_APP_BACKEND_URL = 'http://localhost:8010';

// Mock axios defaults
jest.mock('axios');

// Mock sonner toast
jest.mock('sonner', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
    warning: jest.fn(),
  },
  Toaster: () => null,
}));

// Mock lucide-react icons
jest.mock('lucide-react', () => ({
  Sparkles: () => <div>Sparkles Icon</div>,
  Moon: () => <div>Moon Icon</div>,
  Sun: () => <div>Sun Icon</div>,
  ChevronRight: () => <div>ChevronRight Icon</div>,
  ChevronLeft: () => <div>ChevronLeft Icon</div>,
  Loader2: () => <div>Loader2 Icon</div>,
  Download: () => <div>Download Icon</div>,
  Upload: () => <div>Upload Icon</div>,
  FolderOpen: () => <div>FolderOpen Icon</div>,
  Trash2: () => <div>Trash2 Icon</div>,
  Settings: () => <div>Settings Icon</div>,
  Play: () => <div>Play Icon</div>,
  Database: () => <div>Database Icon</div>,
  CheckCircle2: () => <div>CheckCircle2 Icon</div>,
  XCircle: () => <div>XCircle Icon</div>,
  AlertCircle: () => <div>AlertCircle Icon</div>,
  Info: () => <div>Info Icon</div>,
  Wand2: () => <div>Wand2 Icon</div>,
  RefreshCw: () => <div>RefreshCw Icon</div>,
  Star: () => <div>Star Icon</div>,
  TrendingUp: () => <div>TrendingUp Icon</div>,
  Lightbulb: () => <div>Lightbulb Icon</div>,
  Eye: () => <div>Eye Icon</div>,
  EyeOff: () => <div>EyeOff Icon</div>,
  ChevronDown: () => <div>ChevronDown Icon</div>,
  FileJson: () => <div>FileJson Icon</div>,
}));
