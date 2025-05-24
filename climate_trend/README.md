# Climate Trend Analysis Workflow

This workflow analyzes long-term temperature trends using NOAA Global Summary of the Month (GSOM) data. It downloads weather station data directly from NOAA servers, calculates temperature anomalies relative to a 1991-2020 baseline, and computes statistical trends using distributed processing with Parsl and TaskVine.

## Data Processing Pipeline

1. **Data Source**: Downloads CSV files directly from https://www.ncei.noaa.gov/data/global-summary-of-the-month/access/
2. **Processing**: Converts temperature data from tenths of degrees to Celsius, calculates monthly anomalies
3. **Analysis**: Computes annual trends using linear regression across multiple weather stations
4. **Output**: Generates `annual_trends_large.csv` with comprehensive climate statistics and visualizations

## Running with Floability

### Local Mode
```bash
floability run --backpack climate_trend
```

### Batch Systems
```bash
# HTCondor
floability run --backpack climate_trend --batch-type condor

# Slurm
floability run --backpack climate_trend --batch-type slurm
```

## Configuration

**Default**: Processes 200 weather stations (suitable for distributed computing)

**For local testing**: Edit the first cell of the notebook and set:
```python
MAX_FILES = 10
```

This reduces processing time for development and testing on local machines.
