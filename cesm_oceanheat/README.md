# Calculate Surface Ocean Heat Content Using CESM2 LENS Data

This backpack is adapted from the original notebook:  
[https://github.com/NCAR/osdf_examples/blob/main/notebooks/cesm_oceanheat.ipynb](https://github.com/NCAR/osdf_examples/blob/main/notebooks/cesm_oceanheat.ipynb)

## Install Floability

To run this backpack with Floability, install the CLI tool by following the instructions here:  
[https://github.com/floability/floability-cli](https://github.com/floability/floability-cli)

## Run This Backpack

To run this backpack on your cluster, use:

```bash
floability run --backpack . --batch-type condor
```

If you're using a `slurm` or `uge` cluster, replace `condor` with `slurm` or `uge`.
See all supported cluster types here:
[https://cctools.readthedocs.io/en/latest/man_pages/vine_factory/](https://cctools.readthedocs.io/en/latest/man_pages/vine_factory/)

This command will start a TaskVine manager on the cluster frontend and deploy scalable workers on the cluster nodes.

