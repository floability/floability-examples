#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Check if conda prefix is set
if [ -z "$CONDA_PREFIX" ]; then
  echo "ERROR: CONDA_PREFIX is not set. Please activate your conda environment."
  exit 1
fi

echo "CONDA_PREFIX is set to $CONDA_PREFIX"

# Set working directory and cleanup function
WORK_DIR=$(mktemp -d)
cleanup() {
  echo "Cleaning up temporary directory: $WORK_DIR"
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

cd "$WORK_DIR"

echo "Cloning cctools repository..."
if ! git clone https://github.com/cooperative-computing-lab/cctools.git; then
  echo "ERROR: Failed to clone cctools repository"
  exit 1
fi

cd cctools

echo "Configuring cctools..."
if ! ./configure --with-base-dir "$CONDA_PREFIX" --prefix "$CONDA_PREFIX"; then
  echo "ERROR: Configure failed"
  exit 1
fi

echo "Building cctools (this may take several minutes)..."
if ! make -j"$(nproc)"; then
  echo "ERROR: Build failed"
  exit 1
fi

echo "Installing cctools..."
if ! make install; then
  echo "ERROR: Installation failed"
  exit 1
fi

echo "Verifying installation..."
if vine_worker --version; then
  echo "TaskVine installation completed successfully!"
else
  echo "WARNING: TaskVine installation may have issues - vine_worker command failed"
  exit 1
fi
