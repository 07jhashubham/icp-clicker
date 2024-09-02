#!/bin/bash

BUILD_TARGET="wasm32-unknown-unknown"
BUILD_DIR="target/$BUILD_TARGET/release"
WASM_FILE="akai_backend.wasm"
DID_FILE="akai_backend.did"
SRC_DIR="src/akai_backend"

if [ -f "$BUILD_DIR/$WASM_FILE" ]; then
  echo "Removing existing WASM file..."
  rm "$BUILD_DIR/$WASM_FILE"
fi

if [ -f "$SRC_DIR/$DID_FILE" ]; then
  echo "Removing existing DID file..."
  rm "$SRC_DIR/$DID_FILE"
fi


echo "Building the project..."
cargo build --release --target "$BUILD_TARGET"

echo "Extracting candid..."
candid-extractor "$BUILD_DIR/$WASM_FILE" > "$DID_FILE"

echo "Moving the generated DID file..."
mv "$DID_FILE" "$SRC_DIR/"

echo "Candids generated successfully."
