import { useState, useEffect } from 'react';
import axios from 'axios';
import { Moon, Sun, Sparkles, ChevronRight, ChevronLeft, Download, Upload, FolderOpen, Trash2, HelpCircle } from 'lucide-react';
import { useTheme } from '../components/ThemeProvider';
import { Button } from '../components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../components/ui/dialog';
import { Badge } from '../components/ui/badge';
import ProjectSetup from '../components/steps/ProjectSetup';
import PromptOptimization from '../components/steps/PromptOptimization';
import EvalGeneration from '../components/steps/EvalGeneration';
import DatasetGeneration from '../components/steps/DatasetGeneration';
import TestExecution from '../components/steps/TestExecution';
import SettingsModal from '../components/SettingsModal';
import { Toaster, toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const steps = [
  { id: 1, name: 'Project Setup', component: ProjectSetup },
  { id: 2, name: 'Optimization', component: PromptOptimization },
  { id: 3, name: 'Eval Generation', component: EvalGeneration },
  { id: 4, name: 'Dataset', component: DatasetGeneration },
  { id: 5, name: 'Test & Results', component: TestExecution },
];

export default function Dashboard() {
  const { theme, setTheme } = useTheme();
  const [currentStep, setCurrentStep] = useState(1);
  const [project, setProject] = useState(null);
  const [promptVersions, setPromptVersions] = useState([]);
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [settings, setSettings] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showOpenProject, setShowOpenProject] = useState(false);
  const [allProjects, setAllProjects] = useState([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [autoSaving, setAutoSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);
  const [sessionId] = useState(() => {
    const stored = localStorage.getItem('athena-session-id');
    if (stored) return stored;
    const newId = `session-${Date.now()}`;
    localStorage.setItem('athena-session-id', newId);
    return newId;
  });

  useEffect(() => {
    loadSettings();
  }, []);

  // Auto-save project when it changes
  useEffect(() => {
    if (project) {
      const autoSaveTimer = setTimeout(() => {
        autoSaveProject();
      }, 2000); // Auto-save 2 seconds after changes

      return () => clearTimeout(autoSaveTimer);
    }
  }, [project]);

  const loadSettings = async () => {
    try {
      const response = await axios.get(`${API}/settings/${sessionId}`);
      setSettings(response.data);
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  const loadAllProjects = async () => {
    setLoadingProjects(true);
    try {
      const response = await axios.get(`${API}/projects`);
      setAllProjects(response.data);
    } catch (error) {
      console.error('Error loading projects:', error);
      toast.error('Failed to load projects');
    } finally {
      setLoadingProjects(false);
    }
  };

  const handleOpenProjectModal = () => {
    loadAllProjects();
    setShowOpenProject(true);
  };

  const handleOpenProject = async (selectedProject) => {
    try {
      setProject(selectedProject);
      
      // Load prompt versions for this project
      const versionsResponse = await axios.get(`${API}/prompt-versions/${selectedProject.id}`);
      setPromptVersions(versionsResponse.data);
      
      if (versionsResponse.data.length > 0) {
        setSelectedVersion(versionsResponse.data[0]);
      }
      
      setCurrentStep(2); // Start at optimization step
      setShowOpenProject(false);
      
      toast.success(`Opened: ${selectedProject.name}`, {
        description: `${versionsResponse.data.length} versions loaded`
      });
    } catch (error) {
      console.error('Error opening project:', error);
      toast.error('Failed to open project');
    }
  };

  const handleDeleteProject = async (projectId, projectName) => {
    if (!window.confirm(`Are you sure you want to delete "${projectName}"? This cannot be undone.`)) {
      return;
    }

    try {
      await axios.delete(`${API}/projects/${projectId}`);
      toast.success('Project deleted');
      
      // Reload projects list
      loadAllProjects();
      
      // If deleted project was currently open, clear it
      if (project?.id === projectId) {
        setProject(null);
        setPromptVersions([]);
        setSelectedVersion(null);
        setCurrentStep(1);
      }
    } catch (error) {
      console.error('Error deleting project:', error);
      toast.error('Failed to delete project');
    }
  };

  const autoSaveProject = async () => {
    if (!project) return;

    setAutoSaving(true);
    try {
      await axios.put(`${API}/projects/${project.id}`, {
        name: project.name,
        use_case: project.use_case,
        requirements: project.requirements
      });
      
      setLastSaved(new Date());
      console.log('Project auto-saved');
    } catch (error) {
      console.error('Auto-save failed:', error);
    } finally {
      setTimeout(() => setAutoSaving(false), 500);
    }
  };

  const handleProjectCreated = (newProject) => {
    setProject(newProject);
    setCurrentStep(2);
    toast.success('Project created and auto-saved!');
  };

  const handleVersionCreated = (version) => {
    setPromptVersions([version, ...promptVersions]);
    setSelectedVersion(version);
    // Auto-save happens automatically via useEffect
  };

  const handleExportProject = async () => {
    if (!project) {
      toast.error('No project to export');
      return;
    }

    try {
      const response = await axios.get(`${API}/projects/${project.id}/export`);
      const exportData = response.data;

      // Create downloadable JSON file
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project.name.replace(/\s+/g, '_')}_export_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success('Project exported successfully!', {
        description: `Exported ${exportData.metadata.total_prompt_versions} versions, ${exportData.metadata.total_eval_prompts} eval prompts, and ${exportData.metadata.total_test_cases} test cases`
      });
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export project');
    }
  };

  const handleImportProject = async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      try {
        const text = await file.text();
        const importData = JSON.parse(text);

        const response = await axios.post(`${API}/projects/import`, importData);

        if (response.data.success) {
          toast.success('Project imported successfully!', {
            description: `Imported ${response.data.imported_items.prompt_versions} versions and ${response.data.imported_items.test_cases} test cases`,
            duration: 5000
          });

          // Optionally reload the imported project
          // You could add logic here to load the newly imported project
        }
      } catch (error) {
        console.error('Import failed:', error);
        toast.error('Failed to import project', {
          description: error.response?.data?.detail || 'Invalid project file'
        });
      }
    };
    input.click();
  };

  const CurrentStepComponent = steps[currentStep - 1].component;

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <Toaster position="top-right" richColors />
      
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <img 
                src="/athena-logo.png" 
                alt="Athena Logo" 
                className="w-12 h-12 object-contain"
              />
              <div>
                <div className="flex items-center space-x-2">
                  <h1 className="text-xl font-bold tracking-tight" data-testid="app-title">Athena</h1>
                  {project && (
                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      {autoSaving ? (
                        <>
                          <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></div>
                          <span>Saving...</span>
                        </>
                      ) : lastSaved ? (
                        <>
                          <div className="w-1.5 h-1.5 bg-green-500 rounded-full"></div>
                          <span>Saved</span>
                        </>
                      ) : null}
                    </div>
                  )}
                </div>
                <p className="text-xs font-medium text-primary">Your Strategic Prompt Architect</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleOpenProjectModal}
                data-testid="open-project-button"
                title="Open Project"
              >
                <FolderOpen className="h-4 w-4 mr-1" />
                Open
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.open('/help.html', '_blank')}
                data-testid="help-button"
                title="Help & Documentation"
              >
                <HelpCircle className="h-4 w-4 mr-1" />
                Help
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(true)}
                data-testid="settings-button"
              >
                Settings
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
                data-testid="theme-toggle"
              >
                {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Stepper */}
      <div className="border-b border-border bg-card/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            {steps.map((step, idx) => (
              <div key={step.id} className="flex items-center flex-1">
                <button
                  onClick={() => project && setCurrentStep(step.id)}
                  disabled={!project && step.id > 1}
                  className={`flex items-center space-x-2 transition-all duration-200 ${
                    currentStep === step.id
                      ? 'text-primary font-semibold'
                      : currentStep > step.id
                      ? 'text-foreground hover:text-primary'
                      : 'text-muted-foreground cursor-not-allowed'
                  }`}
                  data-testid={`step-${step.id}-button`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-200 ${
                      currentStep === step.id
                        ? 'bg-primary text-primary-foreground shadow-lg'
                        : currentStep > step.id
                        ? 'bg-primary/20 text-primary'
                        : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {step.id}
                  </div>
                  <span className="hidden md:inline text-sm">{step.name}</span>
                </button>
                {idx < steps.length - 1 && (
                  <div className="flex-1 h-0.5 mx-4 bg-border" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="fade-in">
          <CurrentStepComponent
            project={project}
            promptVersions={promptVersions}
            selectedVersion={selectedVersion}
            settings={settings}
            onProjectCreated={handleProjectCreated}
            onVersionCreated={handleVersionCreated}
            onNextStep={() => setCurrentStep(currentStep + 1)}
            onPrevStep={() => setCurrentStep(currentStep - 1)}
            sessionId={sessionId}
          />
        </div>
      </main>

      {/* Navigation */}
      {project && (
        <div className="fixed bottom-8 right-8 flex space-x-2">
          {currentStep > 1 && (
            <Button
              onClick={() => setCurrentStep(currentStep - 1)}
              variant="outline"
              size="lg"
              className="button-hover shadow-lg"
              data-testid="prev-step-button"
            >
              <ChevronLeft className="w-4 h-4 mr-2" />
              Previous
            </Button>
          )}
          {currentStep < 5 && (
            <Button
              onClick={() => setCurrentStep(currentStep + 1)}
              size="lg"
              className="button-hover shadow-lg"
              data-testid="next-step-button"
            >
              Next
              <ChevronRight className="w-4 h-4 ml-2" />
            </Button>
          )}
        </div>
      )}

      {/* Settings Modal */}
      <SettingsModal
        open={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        sessionId={sessionId}
        onSettingsUpdated={(newSettings) => {
          setSettings(newSettings);
          setShowSettings(false);
        }}
      />

      {/* Open Project Modal */}
      <Dialog open={showOpenProject} onOpenChange={setShowOpenProject}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <FolderOpen className="w-5 h-5 text-primary" />
              <span>Open Project</span>
            </DialogTitle>
            <DialogDescription>
              Load a previous project or create a new one
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 mt-4">
            <div className="grid grid-cols-3 gap-2">
              <Button
                onClick={() => {
                  setShowOpenProject(false);
                  setCurrentStep(1);
                }}
                variant="default"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Create New
              </Button>
              <Button
                onClick={() => {
                  setShowOpenProject(false);
                  handleImportProject();
                }}
                variant="outline"
              >
                <Upload className="w-4 h-4 mr-2" />
                Import
              </Button>
              {project && (
                <Button
                  onClick={() => {
                    setShowOpenProject(false);
                    handleExportProject();
                  }}
                  variant="outline"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              )}
            </div>

            <div className="border-t pt-4">
              <h3 className="font-semibold text-sm mb-3">Existing Projects</h3>
              
              {loadingProjects ? (
                <div className="text-center py-8 text-muted-foreground">
                  Loading projects...
                </div>
              ) : allProjects.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <FolderOpen className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>No projects yet</p>
                  <p className="text-xs mt-1">Create your first project to get started</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {allProjects.map((proj) => (
                    <div
                      key={proj.id}
                      className={`flex items-start justify-between p-4 border rounded-lg hover:border-primary/50 transition-colors cursor-pointer ${
                        project?.id === proj.id ? 'border-primary bg-primary/5' : 'border-border'
                      }`}
                      onClick={() => handleOpenProject(proj)}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="font-medium truncate">{proj.name}</h4>
                          {project?.id === proj.id && (
                            <Badge variant="default" className="text-xs">Current</Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                          {proj.use_case || 'No description'}
                        </p>
                        <div className="flex items-center space-x-3 text-xs text-muted-foreground">
                          <span>
                            Created: {new Date(proj.created_at).toLocaleDateString()}
                          </span>
                          {proj.updated_at && (
                            <span>
                              Updated: {new Date(proj.updated_at).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center space-x-1 ml-4">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={async (e) => {
                            e.stopPropagation();
                            try {
                              const response = await axios.get(`${API}/projects/${proj.id}/export`);
                              const exportData = response.data;
                              const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = `${proj.name.replace(/\s+/g, '_')}_export_${Date.now()}.json`;
                              document.body.appendChild(a);
                              a.click();
                              document.body.removeChild(a);
                              URL.revokeObjectURL(url);
                              toast.success(`Exported: ${proj.name}`);
                            } catch (error) {
                              console.error('Export failed:', error);
                              toast.error('Export failed');
                            }
                          }}
                          title="Export Project"
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteProject(proj.id, proj.name);
                          }}
                          className="text-destructive hover:text-destructive"
                          title="Delete Project"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex justify-end pt-4 border-t">
              <Button variant="outline" onClick={() => setShowOpenProject(false)}>
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}