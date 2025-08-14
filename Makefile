# Makefile for BentoML workflow
# Usage:
#   make build DEPLOYMENT_NAME=my-service
#   make deploy DEPLOYMENT_NAME=my-service
#   make update DEPLOYMENT_NAME=my-service
#   make containerize
#   make push

# Variables
SERVICE_NAME := service:MyService
DEPLOYMENT_NAME ?= my-bento-deployment
BENTO_TAG := $(shell bentoml build --print-tag $(SERVICE_NAME) 2>/dev/null || echo "")

.PHONY: build deploy update containerize push clean

# Build Bento
build:
	bentoml build $(SERVICE_NAME)

# Deploy to BentoCloud
deploy:
ifndef DEPLOYMENT_NAME
	$(error DEPLOYMENT_NAME is not set. Use `make deploy DEPLOYMENT_NAME=my-service`)
endif
	bentoml deploy $(BENTO_TAG) -n $(DEPLOYMENT_NAME)

# Update an existing deployment
update:
ifndef DEPLOYMENT_NAME
	$(error DEPLOYMENT_NAME is not set. Use `make update DEPLOYMENT_NAME=my-service`)
endif
	bentoml deployment update --bento $(BENTO_TAG) $(DEPLOYMENT_NAME)

# Containerize Bento
containerize:
	bentoml containerize $(BENTO_TAG)

# Push Bento to BentoCloud
push:
	bentoml push $(BENTO_TAG)

# Clean build artifacts
clean:
	rm -rf build dist .bento
