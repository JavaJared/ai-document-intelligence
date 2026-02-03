# Terraform configuration for deploying AI Document Intelligence System on AWS
# Includes Lambda, S3, API Gateway, and supporting infrastructure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "ai-doc-intel"
}

# S3 bucket for document storage
resource "aws_s3_bucket" "documents" {
  bucket = "${var.project_name}-documents-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-documents"
    Environment = var.environment
    Project     = var.project_name
  }
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket for vector index storage
resource "aws_s3_bucket" "vector_index" {
  bucket = "${var.project_name}-vector-index-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-vector-index"
    Environment = var.environment
    Project     = var.project_name
  }
}

# DynamoDB table for metadata
resource "aws_dynamodb_table" "document_metadata" {
  name           = "${var.project_name}-metadata-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "document_id"
  
  attribute {
    name = "document_id"
    type = "S"
  }
  
  attribute {
    name = "uploaded_at"
    type = "N"
  }
  
  global_secondary_index {
    name            = "UploadedAtIndex"
    hash_key        = "uploaded_at"
    projection_type = "ALL"
  }
  
  tags = {
    Name        = "${var.project_name}-metadata"
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM role for Lambda execution
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-lambda-role"
    Environment = var.environment
  }
}

# IAM policy for Lambda
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy-${var.environment}"
  role = aws_iam_role.lambda_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*",
          aws_s3_bucket.vector_index.arn,
          "${aws_s3_bucket.vector_index.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.document_metadata.arn,
          "${aws_dynamodb_table.document_metadata.arn}/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Lambda function for document processing
resource "aws_lambda_function" "document_processor" {
  filename         = "lambda_deployment.zip"
  function_name    = "${var.project_name}-processor-${var.environment}"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "lambda_handler.process_document"
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 2048
  
  environment {
    variables = {
      ENVIRONMENT         = var.environment
      S3_BUCKET          = aws_s3_bucket.documents.id
      DYNAMODB_TABLE     = aws_dynamodb_table.document_metadata.name
      VECTOR_INDEX_BUCKET = aws_s3_bucket.vector_index.id
    }
  }
  
  tags = {
    Name        = "${var.project_name}-processor"
    Environment = var.environment
  }
}

# Lambda function for query processing
resource "aws_lambda_function" "query_processor" {
  filename         = "lambda_deployment.zip"
  function_name    = "${var.project_name}-query-${var.environment}"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "lambda_handler.process_query"
  runtime         = "python3.9"
  timeout         = 60
  memory_size     = 2048
  
  environment {
    variables = {
      ENVIRONMENT         = var.environment
      VECTOR_INDEX_BUCKET = aws_s3_bucket.vector_index.id
      OPENAI_API_KEY     = var.openai_api_key
    }
  }
  
  tags = {
    Name        = "${var.project_name}-query"
    Environment = var.environment
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "main" {
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PUT", "DELETE"]
    allow_headers = ["*"]
  }
  
  tags = {
    Name        = "${var.project_name}-api"
    Environment = var.environment
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${var.project_name}-${var.environment}"
  retention_in_days = 14
  
  tags = {
    Name        = "${var.project_name}-logs"
    Environment = var.environment
  }
}

# Outputs
output "s3_document_bucket" {
  description = "S3 bucket for documents"
  value       = aws_s3_bucket.documents.id
}

output "s3_vector_index_bucket" {
  description = "S3 bucket for vector index"
  value       = aws_s3_bucket.vector_index.id
}

output "dynamodb_table" {
  description = "DynamoDB table for metadata"
  value       = aws_dynamodb_table.document_metadata.name
}

output "api_endpoint" {
  description = "API Gateway endpoint"
  value       = aws_apigatewayv2_api.main.api_endpoint
}

# Variable for OpenAI API key (should be set via environment or tfvars)
variable "openai_api_key" {
  description = "OpenAI API key for LLM integration"
  type        = string
  sensitive   = true
  default     = ""
}
