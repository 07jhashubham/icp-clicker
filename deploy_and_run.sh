#!/bin/bash

# Exit immediately if a command exits with a non-zero status

# Deploy akai_frontend
echo "Deploying akai_frontend..."
dfx deploy akai_frontend --playground

# Get the Canister ID of akai_frontend
CANISTER_ID=$(dfx canister id deployer --playground)

# Update canister settings to add a controller
echo "Updating canister settings to add a controller..."
dfx canister update-settings "$CANISTER_ID" --add-controller "$CANISTER_ID --playground"

# Navigate to src/uploader directory
echo "Navigating to src/uploader directory..."
cd src/uploader

# Run cargo
echo "Running cargo..."
cargo run -- $CANISTER_ID