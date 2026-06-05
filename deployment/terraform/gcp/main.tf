terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "api" {
  location      = var.region
  repository_id = "${var.project_name}-api"
  format        = "DOCKER"
}

resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_id}-${var.project_name}-artifacts"
  location      = var.region
  force_destroy = false
  versioning { enabled = true }
}

resource "google_cloud_run_v2_service" "api" {
  name     = "${var.project_name}-api"
  location = var.region
  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.project_name}-api/api:latest"
      ports { container_port = 8000 }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
}

output "cloud_run_url"     { value = google_cloud_run_v2_service.api.uri }
output "artifact_registry" { value = google_artifact_registry_repository.api.id }
