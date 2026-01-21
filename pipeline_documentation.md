# 📘 Ubiquitous Data-Driven Emission Framework — Detailed Documentation
### *Code Structure, Execution Pipeline, and Inter-Module Dependencies*

---

## 🔁 Complete Pipeline Overview

### Step 1 — Visual AI Processing  
**Scripts:** `0.0_image2video.py`, `1.0_camera2traffic.py`, `1.1`, `1.2`, `1.3`  
→ Produces hourly vehicle counts, fleet composition, and signal cycles.  
**Outputs:** `camera_counts.csv`, `signal_cycles.pkl`, `vehicle_mix.csv`

### Step 2 — Dynamic Traffic Simulation  
**Scripts:** `2.0_read_demand.py`, `2.1_dtalite_run.py`, `2.2_dtalite_analysis.py`  
→ Generates link-level traffic flow and speed from OD data.  
**Outputs:** `link_speed_hourly.csv`

### Step 3 — MOVES-Matrix Emission Estimation  
**Scripts:** `3.0_moves_prepare_link.py`, `3.1_moves_matrix.py`, `3.2_emission_results.py`  
→ Computes hourly link-level emissions (CO₂, NOx, CO, PM₂.₅).  
**Outputs:** `emission_link_hourly.csv`, `emission_summary.pdf`

### Step 4 — Policy Evaluation  
**Scripts:** `4.0_scenario_moves_prepare_link.py`, `4.1_scenario_results.py`  
→ Re-runs emission modeling under various demand or fleet scenarios.  
**Outputs:** Scenario impact tables and visualizations.

---

## 🧩 1. VISUAL MODULE (`1-visual/`)
This module extracts **vehicle activity and signal data** from city traffic cameras.  
It is responsible for converting raw video streams into structured traffic counts and signal timing inputs for the DTA simulation.

### **1.0_image2video.py**
**Purpose:** Download and compile images from the NYC DOT live camera API into playable video clips.  
**Main Steps:**
- Access camera image URLs from the NYC Open API.  
- Download sequential images (default: 600 per camera).  
- Compile frames into `.mp4` videos at 1 FPS using OpenCV.  
- Supports concurrent downloads using thread pools.  
**Key Outputs:**  
`NY_Video/{camera_id}.mp4` — one compiled video per camera.

---

### **0.1_binglabeling.py**
**Purpose:** Semi-automatic labeling assistant for object detection training (vehicle bounding boxes).  
**Functionality:**
- Fetch satellite/street map tiles via Bing API.  
- Generates labeling images aligned with camera FOVs.  
- Supports manual or semi-automatic annotation tools.  
**Output:** Labeled image datasets for YOLO/EfficientNet retraining.

---

### **0.2_vehicle_classification.py**
**Purpose:** Vehicle classification using EfficientNet-V2 pretrained on ImageNet / fine-tuned dataset.  
**Workflow:**
- Preprocess cropped vehicle images (224×224).  
- Predict vehicle categories (car, bus, truck, motorcycle, etc.).  
- Save results to `.csv` for downstream aggregation.  
**Output:** Vehicle classification table (`vehicle_labels.csv`).

---

### **1.0_camera2traffic.py**
**Purpose:** Detect, track, and count vehicles per camera using YOLOv8 + EfficientNet-V2.  
**Main Logic:**
1. Load YOLO model for object detection (`conf=0.3`).  
2. Load EfficientNet for vehicle type classification.  
3. Track vehicles across frames, using counting lines (0.8, 0.7, 0.6, 0.5 fractions of image height).  
4. Log entries and exits when vehicles cross the line.  
5. Save per-frame count summaries.

**Key Inputs:**  
- Video folder: `Video_Process/NY_Video/`  
- Camera list: `cams_need.csv`  
- Model files: `YOLOv8 weights`, `EfficientNet model`

**Key Outputs:**  
- `Results2/{camera_id}_counts.csv`  
- `After2/{camera_id}_annotated.mp4` (optional)

---

### **1.1_camera_traffic_analysis.py**
**Purpose:** Aggregate and analyze camera counts across time and space.  
**Functions:**
- Compute hourly/daily volume profiles.  
- Merge with spatial metadata (camera coordinates).  
- Plot diurnal volume patterns and vehicle-type ratios.  
**Output:** `traffic_analysis_summary.csv`, time-series plots.

---

### **1.2_camera_signal_analysis.py**
**Purpose:** Detect traffic signals and estimate timing cycles for signalized intersections.  
**Workflow:**
- Load OSM shapefile of signal locations and road links.  
- Detect intersections of link pairs near signal coordinates.  
- Estimate red/green cycle lengths via motion patterns or heuristic time-series analysis.  
- Join signal metadata to the OSM road network.

**Inputs:**  
`osmdta_ritis.shp`, `gis_osm_traffic_free_1.shp`  

**Outputs:**  
`osmdta_ritis_signal.pkl` — link-level dataset with signal flags and timing.

---

### **1.3_camera_vehicle_analysis.py**
**Purpose:** Analyze vehicle-type composition per camera.  
**Functions:**
- Aggregate classification outputs.  
- Compute daily/hourly shares of cars, trucks, buses, motorcycles.  
- Produce fleet-mix statistics for MOVES input calibration.  
**Outputs:** `camera_vehicle_composition.csv`, bar charts.

---

## 🚦 2. DTA MODULE (`2-dta/`)
This module runs **network-level traffic simulation** and builds the link-based mobility foundation for emission modeling.

### **2.0_read_demand.py**
**Purpose:** Read and clean raw OD matrices or mobility demand tables.  
- Convert raw mobility OD data (from phone or survey).  
- Normalize temporal distribution.  
- Prepare `demand_AM.txt`, `demand_PM.txt`, etc. for DTALite.  
**Output:** Cleaned OD tables.

---

### **2.1_dtalite_run.py**
**Purpose:** Run DTALite simulation using OSM-based Manhattan network.  
**Key Functions:**
- Load OSM shapefile and link nodes.  
- Assign TAZ (traffic analysis zones) using Census tract IDs.  
- Execute DTALite for multi-period runs (`am`, `pm`, `md`, `nt`).  
- Export link-level flow, speed, and density data.

**Inputs:**  
`Simulation_osm/network/`, `OD demand tables`

**Outputs:**  
`link_performance.csv`, `link_speed_hourly.csv`

---

### **2.2_dtalite_analysis.py**
**Purpose:** Post-process DTALite outputs and align them with observed camera data.  
**Main Steps:**
- Merge simulated link volumes with observed camera counts.  
- Compare baseline vs. event days (snowstorm, hurricane, etc.).  
- Fit VSD (volume-speed-density) curves.  
- Export link-level average speeds per hour.

**Outputs:**  
`vsd_fits.pkl`, `link_speed_cleaned.csv`, visual diagnostics.

---

### **2.3_VSD_analysis.py**
**Purpose:** Quantify fundamental relationships between flow, speed, and density.  
**Logic:**
- Use regression (nonlinear least squares) to fit curves.  
- Validate link-specific fundamental diagrams.  
- Plot fitted relationships for calibration.

---

## 🌍 3. MOVES-MATRIX MODULE (`3-moves-matrix/`)
This stage translates link-level speeds into **link–hour emission factors** using the **MOVES-Matrix** framework.

### **3.0_moves_prepare_link.py**
**Purpose:** Generate MOVES-Matrix-ready input files.  
**Workflow:**
1. Combine DTALite link flows with signal data.  
2. Create hourly driving cycles for each link.  
3. Generate:
   - `link.csv` — metadata for each road link  
   - `drivingCycle.csv` — second-by-second speeds  
   - `sourceTypeDistribution.csv` — fleet composition  
4. Define link metadata (county, zone, road type, grade).  
5. Write 24 hourly input batches for MOVES-Matrix.

**Outputs:**  
`MOVES/input_hourXX/` — hourly folder with required CSVs.

---

### **3.1_moves_matrix.py**
**Purpose:** Interface with MOVES-Matrix engine (developed by Georgia Tech).  
**Functionality:**
- Calls the MOVES-Matrix executable via batch mode.  
- Reads pre-generated county-level emission factors (by Georgia Tech).  
- Produces link-hour emission estimates for each pollutant.  
**Output:**  
`MOVES/output/emission_link_hourly.csv`

---

### **3.2_emission_results.py**
**Purpose:** Aggregate and visualize MOVES-Matrix results.  
**Steps:**
- Merge emission results across hours and pollutants.  
- Compare with/without-signal cases.  
- Create emission maps, temporal charts, and pollutant summaries.  
**Outputs:**  
`emission_summary.csv`, `pollutant_spatial_distribution.pdf`.

---

## 📊 4. SCENARIO MODULE (`4-scenario/`)
This module evaluates **transportation and policy scenarios** and quantifies their emission impacts.

### **4.0_scenario_moves_prepare_link.py**
**Purpose:** Generate new MOVES-Matrix input files for each policy scenario.  
**Scenario Types:**
- **Peak-hour shift (10–30%)** — redistribute traffic from 7–8 & 16–17 to shoulders (6,9,15,18).  
- **Mode shift (10–30%)** — transfer car volume to public transit.  
**Outputs:**  
`MOVES/input_mode10`, `input_peak20`, etc.

---

### **4.1_scenario_results.py**
**Purpose:** Compare scenario-based emissions to baseline.  
**Steps:**
- Read MOVES-Matrix outputs per scenario.  
- Calculate emission differences (% change).  
- Aggregate results at hourly and citywide scales.  
- Produce:
  - Line plots of hourly change  
  - Bar plots of total reductions  
  - Maps of spatial impact

**Outputs:**  
`E_mode_shift.pdf`, `E_peak_shift.pdf`, `E_remote_work.pdf`, `E_congestionprice.pdf`.

---



