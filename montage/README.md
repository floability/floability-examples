# Montage Backpack for Floablity

This repository contains a backpack version of the Montage workflow—a toolkit for constructing science-grade astronomical image mosaics. The backpack captures the entire workflow, including code, data description, required software dependencies and computing resources. This backpack can be deployed using a single command with [floability](https://github.com/floability/floability-cli).

```
floability execute --backpack . --prefer-python
```

## 🔭 What is Montage?

> Montage is a portable software toolkit for constructing custom, science-grade mosaics by composing multiple astronomical images. The mosaics preserve astrometry (source positions) and photometry (source intensities), making them suitable for scientific analysis.

Montage supports user-defined mosaic parameters including coordinate system, spatial sampling, projection, and region of interest. It can run on desktops, clusters, and grids using both single and multi-processor setups.



The original Montage tutorial (detailed in the Montage documentation) demonstrates how to:

- Create metadata tables for input images.

- Generate a FITS header that defines the mosaic's spatial footprint.

- Reproject input images according to the header.

- Combine the reprojected images into a mosaic.

- Produce visualization outputs (PNG).

📄 **Citation:**

> Jacob, Joseph C., et al. "Montage: a grid portal and software toolkit for science-grade astronomical image mosaicking." International Journal of Computational Science and Engineering 4.2 (2009): 73-87. 
> [Link to Paper](https://arxiv.org/abs/1005.4454)

## Usage
Once you have installed [floability](https://github.com/floability/floability-cli), you can run the Montage backpack with the following command:

```bash
floability execute --backpack . --prefer-python
```

The "--backpack" flag specifies the path to the backpack directory.

This command will:
- Fetch the required data files from the specified in the data/data.yml file.
- Create a conda environment with the required dependencies and create tar.gz file of the environment.
- Start a factory to provision workers for the workflow as specified in the compute/compute.yml file.
- Execute the workflow scripted in the workflow/montage.py file.

After the workflow is complete, you will find a png file in workflow directory. The png file is a mosaic of the 91 2MASS K-band images centered on M17. The image will look like this:

![example-output.png](workflow/example-output.png)

## Credits
- [Original Montage Tutorial](http://montage.ipac.caltech.edu/docs/first_mosaic_tutorial.html) - The backpack is based on the Montage tutorial that walks through building a mosaic of 91 2MASS K-band images centered on M17.

- [Colin Thomas](https://github.com/colinthomas-z80/task-group-artifact/tree/main/montage_application) - Collin converted the montage tutorial to a distributed workflow with [Parsl](https://parsl-project.org/) and [TaskVine](https://ccl.cse.nd.edu/software/taskvine/). 