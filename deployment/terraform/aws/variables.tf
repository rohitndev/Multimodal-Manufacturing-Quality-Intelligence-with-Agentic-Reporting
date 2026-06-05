variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for the inspection stack."
}

variable "project_name" {
  type        = string
  default     = "quality-inspection"
  description = "Prefix applied to every created resource."
}
