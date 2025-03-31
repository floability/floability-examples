#!/bin/bash

# Check if conda prefix is set
if [ -z "$CONDA_PREFIX" ]; then
  echo "CONDA_PREFIX is not set. Please activate your conda environment."
  exit 1
fi

echo "CONDA_PREFIX is set to $CONDA_PREFIX"


sleep 5

git clone https://github.com/Caltech-IPAC/Montage.git

cd Montage

./configure --prefix "$CONDA_PREFIX"

make

make install

# Clean up (remove the cloned repository)
cd ..
#rm -rf Montage

echo "Montage installation complete."
