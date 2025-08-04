# Floability Examples


This repository contains a collection of example [floability](https://github.com/floability/floability-cli) backpacks. Backpacks are self-contained and portable representations of scientific workflows.

# Quick Start

Floability is available on conda-forge. To install it, use the following command:

```
conda install -c conda-forge floability
```

Now you are ready to run the floability command-line tool. You can run examples as backpacks like:

```
floability run --backpack distributed_image_convolution
```

To deploy the workers on a batch system, we can use the `--batch-type` flag. This will submit the workers to a job scheduler like HTCondor, UGE or SLURM. For example:

```
floability run --backpack distributed_image_convolution --batch-type condor
```

# Site-Specific Instructions

Although the general instructions mentioned above should work in most HPC clusters, every HPC site has different settings, permissions, and configurations. You may need to adjust some settings when running floability at different sites. Here are site-specific instructions for sites where we have tested floability: 

## ND CRC

The Notre Dame Center for Research Computing (CRC) works without any extra arguments. You can simply run:

```
floability run --backpack distributed_image_convolution --batch-type condor
```

## OSPool

OSPool (Open Science Pool) is a distributed high-throughput computing platform that provides access to computing resources across multiple institutions. Floability is built on top of [TaskVine](https://cctools.readthedocs.io/en/latest/taskvine/), which requires TCP connections between the manager (running on the login node) and workers (running on cluster nodes). However, OSPool does not allow open TCP connections between login nodes and cluster nodes by default.

### Prerequisites

**Step 1: Port Access Permission**
You need special permission from OSPool administrators to open sockets on specific ports (e.g., 502-510). These ports must run in `authbind` mode. Contact OSPool support to request this permission.

**Step 2: Storage Configuration**
OSPool does not allow peer transfer between cluster nodes, so you need to disable peer transfer in TaskVine. This requires large storage on the manager so it can store all the data for workers to access. Since the `/home` partition doesn't have enough space, use the OSPool data volume.

Create a directory in your data volume (e.g., `vine-data`):
```
# For ap40 access point, the full path would be:
mkdir -p /ospool/ap40/data/<USERNAME>/vine-data
```

### Setup Instructions

1. **Create the floability base directory**
   
   Create a directory in your home directory to store floability logs and conda environment tarballs for reuse:
   ```
   mkdir -p ~/floability-base-dir
   ```

2. **Create conda environment**
   
   Make sure you have conda installed and updated. We recommend using conda 25 or later:
   ```
   conda create -n floability-env -c conda-forge python=3.13 floability -y
   ```

3. **Activate the environment and navigate to the repository**
   
   ```
   conda activate floability-env
   cd floability-examples
   ```

### Running Floability on OSPool

Use the following command to run an example (using the `cesm_oceanheat` backpack):

```
authbind --deep floability run --backpack cesm_oceanheat \
  --base-dir ~/floability-base-dir \
  --batch-type condor \
  --manager-ports 502,510 \
  --env-vars="VINE_RUN_INFO_DIR=/ospool/ap40/data/<USERNAME>/vine-data,DISABLE_PEER_TRANSFER=true"
```

**Important:** Replace `<USERNAME>` with your actual username.

#### Command Explanation:
- `authbind --deep`: Allows binding to the specified ports with proper permissions
- `--base-dir`: Specifies where floability stores logs and conda environment tarballs
- `--manager-ports`: Specifies the port range approved by OSPool administrators
- `--env-vars`: Sets environment variables:
  - `VINE_RUN_INFO_DIR`: Points to the data volume directory for large storage
  - `DISABLE_PEER_TRANSFER=true`: Disables peer transfer between cluster nodes

### Expected Output

Once floability successfully starts, you will see output similar to this:

```
[provision] vine_factory stdout: /home/mdsaiful.islam/floability-base-dir/floability_run_20250804_131640_745627/vine_factory.stdout
[floability] Starting JupyterLab...
...

[jupyter] JupyterLab is running on port 8890 on 128.105.68.62.

    You can access it using one of the following URLs:
    local:  http://localhost:8890/lab/?token=5b1cf4550b998f6d065748c085dff5a261dfdc018afa3260
    remote: http://128.105.68.62:8890/lab/?token=5b1cf4550b998f6d065748c085dff5a261dfdc018afa3260

    If you are on a remote machine and it doesn't allow direct access to the port, you can create an SSH tunnel:

    1. Open a terminal and run the following command:
       ssh -L localhost:8890:localhost:8890 mdsaiful.islam@128.105.68.62

    2. Open a web browser and enter the following URL:
       http://localhost:8890/lab/?token=5b1cf4550b998f6d065748c085dff5a261dfdc018afa3260

[jupyter] You can access full jupyterlab log at /home/mdsaiful.islam/floability-base-dir/floability_run_20250804_131640_745627/jupyterlab.stdout
```

### Accessing JupyterLab

1. **Create SSH tunnel**: In another terminal, create an SSH tunnel as mentioned in the output message. Note that floability shows the IP address of the login node by default. If you use a hostname to set up your SSH keys, replace the IP address with the hostname in the SSH command.

2. **Open JupyterLab**: Once you create the SSH tunnel, copy the URL and paste it into your browser. You should see the JupyterLab interface.

### Monitoring Progress

To monitor the progress of your workflow, you can check the `vine_factory.stdout` log file:
```
tail -f /path/to/vine_factory.stdout
```

