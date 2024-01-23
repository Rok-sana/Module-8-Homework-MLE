#!/bin/bash
set -e

function create_directory() {
    local directory=$1
    [ ! -d "$directory" ] && mkdir "$directory"
}

function build_and_run_docker() {
    local dockerfile=$1
    local image_name=$2
    local build_arg1=$3
    local build_arg2=$4

    docker build -f "$dockerfile" --build-arg settings_name="$build_arg1" --build-arg model_name="$build_arg2" -t "$image_name" .
    container_id=$(docker run -d "$image_name")

    docker wait "$container_id"
    docker cp "$container_id:/app/models/sample_model.keras" ./models
    docker cp "$container_id:/app/data/" .
}

# Create directories if they don't exist
create_directory "data"
create_directory "models"

# Build and run training Docker container
build_and_run_docker "./training/Dockerfile" "training_image" "settings.json" ""

# Build and run inference Docker container
docker build -f ./inference/Dockerfile --build-arg model_name=sample_model.keras --build-arg settings_name=settings.json -t inference_image .
container_id=$(docker run -d inference_image)
docker wait $container_id
docker cp $container_id:/app/results .