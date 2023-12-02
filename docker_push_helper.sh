#!/bin/bash

# Variables
REGISTRY_NAME="pushkarmlops23.azurecr.io"
DEPENDENCY_IMAGE_NAME="dependencydocker"
FINAL_IMAGE_NAME="mlops_major"
DEPENDENCY_IMAGE_TAG="$REGISTRY_NAME/$DEPENDENCY_IMAGE_NAME:latest"
FINAL_IMAGE_TAG="$REGISTRY_NAME/$FINAL_IMAGE_NAME:latest"

# Azure login
az login

az acr login --name $REGISTRY_NAME

docker tag $DEPENDENCY_IMAGE_NAME $DEPENDENCY_IMAGE_TAG

docker tag $FINAL_IMAGE_NAME $FINAL_IMAGE_TAG

docker push $DEPENDENCY_IMAGE_TAG

docker push $FINAL_IMAGE_TAG

az acr repository list --name $REGISTRY_NAME --output table
