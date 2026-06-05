variable "project_id" {
  type        = string
  description = "GCP project ID."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP region."
}

variable "project_name" {
  type        = string
  default     = "quality-inspection"
  description = "Prefix applied to every created resource."
}
