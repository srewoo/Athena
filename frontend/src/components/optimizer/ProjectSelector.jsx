/**
 * Project Selector Component
 * Handles loading, creating, and switching between projects
 */
import React from "react";
import { FolderOpen, Plus, Trash2, Pencil, RefreshCw } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { useToast } from "../../hooks/use-toast";

const ProjectSelector = ({
  open,
  onOpenChange,
  projects,
  isLoading,
  currentProjectId,
  onLoadProject,
  onNewProject,
  onEditProject,
  onDeleteProject,
  onRefresh
}) => {
  const { toast } = useToast();

  const formatDate = (dateString) => {
    if (!dateString) return "Unknown";
    const date = new Date(dateString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleDelete = async (project, e) => {
    e.stopPropagation();

    const confirmed = window.confirm(
      `Are you sure you want to delete "${project.project_name || project.name}"?\n\nThis action cannot be undone.`
    );

    if (confirmed) {
      onDeleteProject(project.id, project.project_name || project.name);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5" />
            Projects
          </DialogTitle>
          <DialogDescription>
            Load an existing project or create a new one
          </DialogDescription>
        </DialogHeader>

        <div className="flex gap-2 mb-4">
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button size="sm" onClick={onNewProject}>
            <Plus className="h-4 w-4 mr-2" />
            New Project
          </Button>
        </div>

        <div className="overflow-y-auto max-h-[400px] space-y-2">
          {isLoading ? (
            <div className="text-center py-8 text-muted-foreground">
              Loading projects...
            </div>
          ) : projects.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No projects found. Create your first project to get started.
            </div>
          ) : (
            projects.map((project) => (
              <Card
                key={project.id}
                className={`cursor-pointer transition-colors hover:bg-accent ${
                  currentProjectId === project.id ? 'border-primary' : ''
                }`}
                onClick={() => onLoadProject(project.id)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium truncate">
                        {project.project_name || project.name}
                      </h4>
                      <p className="text-sm text-muted-foreground truncate">
                        {project.use_case || project.requirements?.use_case || "No use case"}
                      </p>
                      <div className="flex gap-4 mt-1 text-xs text-muted-foreground">
                        <span>Updated: {formatDate(project.updated_at)}</span>
                        {project.system_prompt_versions && (
                          <span>v{project.system_prompt_versions.length}</span>
                        )}
                        {project.has_results && (
                          <span className="text-green-600">Has results</span>
                        )}
                      </div>
                    </div>

                    <div className="flex gap-1 ml-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={(e) => {
                          e.stopPropagation();
                          onEditProject(project, e);
                        }}
                        title="Edit"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive hover:text-destructive"
                        onClick={(e) => handleDelete(project, e)}
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProjectSelector;
