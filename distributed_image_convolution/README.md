# Distributed Image Convolution 
This notebook slices a large image into tiles and sends each tile to TaskVine workers, where a selected convolution kernel is applied with the im2col × GEMM method. Tiles return filtered, are stitched back together, and the heavy compute happens in parallel on the cluster.

## Quick start

### Launch Locally

```bash
floability run --backpack .
```
### Launch the backpack on any Condor cluster

```bash
floability run --backpack . --batch-type condor
```
