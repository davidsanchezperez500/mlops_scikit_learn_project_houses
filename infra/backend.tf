terraform {
  backend "gcs" {
    bucket = "mlops-terraform-state-578457"
    prefix = "scikit-learn-project-houses"
  }
}
