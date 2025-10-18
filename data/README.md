
DT-BIM Experimental Dataset
===========================

This dataset provides synthetic data approximating the structure described in the experimental section of the paper titled "Digital Twin-Driven BIM Simulation: Enhancing Construction Process Optimization with Real-Time Data".  
The purpose of the dataset is to support experiments on multi-source data fusion, optimization algorithms and digital twin-driven BIM simulations for construction projects.

### Files Included

- **tasks.csv** – Metadata for 764 construction tasks across three subsystems. Columns:
  - `task_id`: Unique task identifier.
  - `subsystem`: Subsystem number (1: foundation and underground, 2: main structural, 3: steel installation).
  - `task_type`: Category of task (e.g. pile, core_tube, steel_beam).
  - `planned_start`, `planned_end`: Planned schedule dates (ISO format).
  - `actual_start`, `actual_end`: Actual schedule dates (ISO format) with random delays.
  - `required_workers`: Estimated number of workers assigned to the task.
  - `required_equipment`: Main equipment type used for the task.

- **progress_series.csv** – Time series of progress for each task sampled weekly. Columns:
  - `task_id`: Identifier linking to tasks.csv.
  - `timestamp`: ISO datetime of the progress record.
  - `progress`: Completion fraction (0 to 1).
  - `progress_rate`: Rate of progress at the sampling time.
  - `progress_accel`: Change in progress rate (acceleration).
  - `quality_score`: Continuous quality score (0 to 1).

- **sensors.csv** – List of 290 physical sensors installed on the construction site. Columns:
  - `sensor_id`: Unique sensor identifier (SHT for temperature/humidity, VIB for vibration, DIS for displacement).
  - `sensor_type`: Type of sensor (`temp_humidity`, `vibration`, `displacement`).

- **sensor_readings.csv** – Sensor readings over 7 days at 30-minute intervals. Each row corresponds to one measurement (multiple measurement types for temp/humidity sensors). Columns:
  - `sensor_id`: Identifier linking to sensors.csv.
  - `timestamp`: ISO datetime of the reading.
  - `measurement_type`: Measurement category (`temperature`, `humidity`, `vibration`, `displacement`).
  - `value`: Recorded value (units depend on the measurement type).

- **environment_data.csv** – Environmental measurements for 12 monitoring zones over the same time period. Columns:
  - `zone_id`: Zone identifier (Zone01 to Zone12).
  - `timestamp`: ISO datetime.
  - `temperature`, `humidity`, `wind_speed`, `air_quality`: Environmental metrics recorded in the zone.

- **resources.csv** – Definition of available resources on the construction site. Columns:
  - `resource_id`: Unique resource identifier.
  - `resource_type`: Type of resource (e.g. excavator, tower_crane, welder).

- **resource_states.csv** – Time series of resource utilization status sampled at the same intervals as sensor readings. Columns:
  - `resource_id`: Identifier linking to resources.csv.
  - `timestamp`: ISO datetime.
  - `status`: Whether the resource is `busy` or `available`.
  - `assigned_task_id`: Task identifier if the resource is in use; empty otherwise.

- **quality_data.csv** – Synthetic quality inspection results for each task at the end of its execution. Columns:
  - `task_id`: Identifier linking to tasks.csv.
  - `inspection_date`: Date of quality inspection.
  - `compressive_strength_mpa`: Measured concrete compressive strength (for concrete-related tasks).
  - `surface_flatness_mm_per_m`: Measured surface flatness (for concrete-related tasks).
  - `weld_quality_score`: Measured quality score for welded connections (for steel tasks).

### Notes

- The data are randomly generated using distributions inspired by the problem description.  
- Time intervals and measurement frequencies are reduced for manageable file sizes. Adjust sampling frequency and durations to suit your experiments.  
- Quality, progress and resource assignment values do not reflect any real construction project. They are provided solely for algorithm development and testing.

