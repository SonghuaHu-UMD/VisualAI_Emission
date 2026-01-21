# 🌍 Ubiquitous Data-Driven Framework for Traffic Emission Estimation and Policy Evaluation
<!-- ### *An Integrated Pipeline Combining Visual AI, Dynamic Traffic Assignment, and MOVES-Matrix–Based Emission Modeling* -->

---

## 🌐 Overview
This repository presents an **end-to-end AI framework** for large-scale, data-driven **traffic emission estimation and policy evaluation**. The system integrates:

- **Visual AI** for vehicle detection and classification from traffic cameras,  
- **Dynamic Traffic Assignment (DTA)** via **DTALite** for network-level traffic simulation, and  
- **MOVES-Matrix** for high-performance emission modeling.

Applied to **~300 cameras** and **millions of mobile phones** in Manhattan, the framework reconstructs fine-grained mobility patterns and quantifies the environmental impacts of major transportation policies such as **congestion pricing, mode shift, departure time shift,** and big events such as **COVID-19, holidays, and extreme weather**.

---

## 🧭 Pipeline Overview

For more detailed explanation, please refer to [pipeline documentation.md](pipeline_documentation.md).

```
├── 1-visual: Camera data crawling, vehicle detection & classification, signal extraction
    ├── 0.0_image2video.py             # Convert NYDOT image feeds to video
    ├── 0.1_binglabeling.py            # Collect training images for vehicle type classification via Bing Image
    ├── 0.2_vehicle_classification.py  # Train deep vehicle type classifiers (EfficientNet)
    ├── 1.0_camera2traffic.py          # Run detection, tracking, and classification on camera footage
    ├── 1.1_camera_traffic_analysis.py # Analyze vehicle volumes extracted from cameras footage across space and time
    ├── 1.2_camera_signal_analysis.py  # Extract traffic signal cycles from cameras footage
    ├── 1.3_camera_vehicle_analysis.py # Vehicle type distribution, confusion matrix, accuracy evaluation

├── 2-dta: OD demand processing, DTALite simulation, and fundamental-diagram calibration
    ├── 2.0_read_demand.py             # Generate OD demand from mobile device data
    ├── 2.1_dtalite_run.py             # Create simulation input files and run DTALite
    ├── 2.2_dtalite_analysis.py        # Parse simulation output (link-level speed and volume)
    ├── 2.3_VSD_analysis.py            # Fit fundamental diagrams (volume-speed-density)

├── 3.0_moves_prepare_link.py      # Prepare MOVES inputs (link speeds, fleet mix)
    ├── 3.0_moves_prepare_link.py      # MOVES-Matrix input generation, link-level emission computation, pollutant aggregation
    ├── 3.1_moves_matrix.py            # Run MOVES-Matrix batch mode
    ├── 3.2_emission_results.py        # Analyze emissions, spatial maps, time variation

├── 4-scenario: Policy and event simulation (mode shift, peak shift, congestion pricing, COVID-19)
    ├── 4.0_scenario_moves_prepare_link.py  # Create hypothetical scenario inputs (mode shift, pricing, etc.)
    ├── 4.1_scenario_results.py             # Evaluate impact of each scenario on emissions
```

---

## ⚖️ Data Accessibility
| Module | Data Required | Publicly Runnable? | Description |
|:--|:--|:--:|:--|
| **1-visual** | Public webcams or videos | ✅ Yes | Fully runnable with open imagery (e.g., [NYC DOT](https://webcams.nyctmc.org/api/cameras/)). |
| **2-dta** | OpenStreetMap network + mobile-phone OD data | ⚠️ Partial | Network setup (From OpenStreetMap) and [DTALite simulation](https://github.com/asu-trans-ai-lab/DTALite/tree/main) are open; OD calibration from proprietary mobility data (e.g., Cuebiq, SafeGraph, NY MPO) is restricted; Fundamental diagram calibration from proprietary traffic flow data (e.g., INRIX, TomTom) is restricted. |
| **3-moves** | DTALite outputs + MOVES-Matrix engine and emission-factor database | ⚠️ Partial | Input-generation scripts are open. [MOVES-Matrix](https://tse.ce.gatech.edu/development-of-moves-matrix/) and **county-level emission-factor matrices** must be obtained separately. |
| **4-scenario** | MOVES-Matrix output tables | ⚠️ Partial | Scenario evaluation and visualization run fully with the outputs of MOVES-Matrix. |

---

## ⚙️ Configurations
The main parameters are listed in a single configuration file, `config.yaml`.

```yaml
# Global paths
  video_raw: ./Video_Process/NY_Video
  video_processed: ./Video_Process/After2
  results_dir: ./Video_Process/Results2
  model_dir: ./Video_Process/data_models
  osm_ritis_shp: ./Shp/osmdta_ritis.shp
  osm_signal_pkl: ./Shp/osmdta_ritis_signal.pkl
  county_shp: ./Shp/US_county_2019.shp
  taz_shp: ./Shp/NYBPM2012_TAZ.shp
  tract_shp: ./Shp/tracts.shp
  moves_input_root: ./MOVES/input

# Spatial:
  crs_projected: EPSG:32618
  crs_lonlat: EPSG:4326
  county_buffer_m: 100
  county_id: 36061

# YOLO & EfficientNet
    yolo_model: yolov8x.pt
    classifier_model: efficientnetv2_rw_t.pkl
    confidence_threshold: 0.5
    class_input_size: 224
    frame_interval: 2, 5  # seconds

# DTALite
    dtalite_exe: ./DTALite/DTALite.exe
    time_periods: [am, md, pm, nt1, nt2]

# MOVES-Matrix
    county_code: 36061
    pollutants: [CO, NOx, CO2, PM2.5]
    meteorology: meteorology_NYC_36061_01.csv
    age_distribution: ageDistribution_2023.csv

# Scenario parameters
    mode_shift: [10, 20, 30]
    peak_shift: [10, 20, 30]
    congestion_weeks: [2, 4, 6, 8]
    events: ['ns', 'nsp', 'nvo', 'nv']
    ablation: ['ns', 'nsp', 'nvo', 'nv']
```

---

## 🔍 Scenario Evaluations

The framework supports evaluating several realistic interventions:

| Scenario Code     | Description                                 |
|-------------------|---------------------------------------------|
| `s_raw`           | Baseline observed travel behavior           |
| `s_mode10/20/30`  | Mode shift to public transit (10–30%)       |
| `s_peak10/20/30`  | Departure time shift from peak hours        |
| `s_cong_2/4/6/8`  | Weeks after NYC congestion pricing launch   |
| 'ns', 'nsp', 'nvo', 'nv'  | Ablation analysis (w/o singal control, average fleet composition, average speed, w/o cameras) |
| 'cd', 'te', 'hf', 'ss'  | Real-world disruptions (COVID, flooding, snowstrom, holidays, etc.) |

---

## ▶️ Quick-Start Workflow
**Python ≥ 3.10**
1. Edit `config.yaml` for your local paths and parameters. 
2. Install Dependencies: `pandas`, `geopandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `shapely`,  
`timm`, `torch`, `fastai`, `ultralytics`, `contextily`, `mapclassify`, `scipy`
3. Install external tools:
    - [DTALite](https://github.com/asu-trans-ai-lab/DTALite) for dynamic traffic simulation
    - Georgia Tech [MOVES-Matrix](https://tse.ce.gatech.edu/development-of-moves-matrix/) for emission modeling
4. Run the full pipeline:

```bash
python 1-visual/0.0_image2video.py
python 1-visual/0.1_binglabeling.py
...
```
---

