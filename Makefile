# Simplified Makefile for BentoML workflow with fixed Bento tag
# Usage:
#   make build
#   make serve
#   make deploy DEPLOYMENT_NAME=my-service

# Variables
SERVICE_NAME := src.predict:FluxLoRAService
BENTO_TAG := flux_lora_service:dznhtjtzdkbc47a6
DEPLOYMENT_NAME ?= my-bento-deployment

.PHONY: build serve deploy

# Build Bento
build:
	bentoml build $(SERVICE_NAME)

# Serve locally
serve:
	bentoml serve $(SERVICE_NAME)

# Deploy to BentoCloud
deploy:
ifndef DEPLOYMENT_NAME
	$(error DEPLOYMENT_NAME is not set. Use `make deploy DEPLOYMENT_NAME=my-service`)
endif
	bentoml deploy $(BENTO_TAG) -n $(DEPLOYMENT_NAME)
