#bento usage file

# Variables
SERVICE_NAME := service:MyService
BENTO_TAG := flux-lora-inference:latest

# Dry-run: Check if service is importable and BentoML config is valid
dry-run:
	bentoml build --print --service $(SERVICE_NAME)

# Build Bento
build:
	bentoml build

# Build Bento and tag
build-tag:
	bentoml build -t $(BENTO_TAG)

# Serve locally (debug mode)
serve:
	bentoml serve $(SERVICE_NAME) --reload

# Serve locally with GPU
serve-gpu:
	bentoml serve $(SERVICE_NAME) --reload --device cuda

# Build Docker image from Bento
docker-build:
	bentoml containerize $(BENTO_TAG)

# Push Docker image to registry
docker-push:
	docker push $(BENTO_TAG)

# Deploy to BentoCloud (requires login & config)
deploy:
	bentoml deploy $(BENTO_TAG) --platform docker

# List Bentos in local store
list:
	bentoml list

# Clean up built Bentos
clean:
	bentoml delete --yes $(BENTO_TAG) || true

# Full workflow: dry-run -> build -> containerize
all: dry-run build docker-build
