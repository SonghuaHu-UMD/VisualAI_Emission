# Ubiquitous Data-Driven Framework for Traffic Emission Estimation and Policy Evaluation

---

## Overview
This repository presents an **end-to-end AI framework** for large-scale, data-driven **traffic emission estimation and policy evaluation**. The system integrates:

- **Visual AI** for vehicle detection and classification from traffic cameras,  
- **Dynamic Traffic Assignment (DTA)** via **DTALite** for network-level traffic simulation, and  
- **MOVES-Matrix** for high-performance emission modeling.

Applied to **~300 cameras** and **millions of mobile phones** in Manhattan, the framework reconstructs fine-grained mobility patterns and quantifies the environmental impacts of major transportation policies such as **congestion pricing, mode shift, departure time shift,** and big events such as **COVID-19, holidays, and extreme weather**.

---

## Pipeline Overview

For more detailed explanation, please refer to [pipeline_documentation.md](pipeline_documentation.md).

```
├── config.py                          # Centralized paths, constants, and parameter definitions
├── utils.py                           # Shared utility functions (network loading, Voronoi, BPR, data parsing)
│
├── 1-visual/                          # Camera data crawling, vehicle detection & classification, signal extraction
│   ├── 0.0_image2video.py             # Convert NYDOT image feeds to video
│   ├── 0.1_binglabeling.py            # Collect training images for vehicle type classification via Bing Image
│   ├── 0.2_vehicle_classification.py  # Train deep vehicle type classifiers (EfficientNet-v2)
│   ├── 1.0_camera2traffic.py          # Run detection, tracking, and classification on camera footage
│   ├── 1.1_camera_traffic_analysis.py # Analyze vehicle volumes across space and time
│   ├── 1.2_camera_signal_analysis.py  # Extract traffic signal cycles from camera footage
│   └── 1.3_camera_vehicle_analysis.py # Vehicle type distribution, confusion matrix, accuracy evaluation
│
├── 2-dta/                             # OD demand processing, DTALite simulation, and fundamental-diagram calibration
│   ├── 2.0_read_demand.py             # Generate OD demand from mobile device data
│   ├── 2.1_dtalite_run.py             # Create simulation input files and run DTALite
│   ├── 2.2_dtalite_analysis.py        # Parse simulation output (link-level speed and volume)
│   └── 2.3_VSD_analysis.py            # Fit fundamental diagrams (volume-speed-density)
│
├── 3-moves/                           # MOVES-Matrix input generation, emission computation, and result analysis
│   ├── 3.0_moves_prepare_link.py      # Prepare MOVES inputs (driving cycles, fleet mix, link metadata)
│   ├── 3.1_moves_matrix.py            # Run MOVES-Matrix batch mode (VSP/opmode for all 13 source types)
│   └── 3.2_emission_results.py        # Analyze emissions: spatial maps, time variation, ablation study
│
├── 4-scenario/                        # Policy and event simulation
│   ├── 4.0_scenario_moves_prepare_link.py  # Create scenario inputs (mode shift, peak shift, congestion pricing)
│   └── 4.1_scenario_results.py             # Evaluate emission impact of each scenario
│
└── data/                              # Intermediate and output data files
```

---

## Shared Modules

| File | Description |
|------|-------------|
| `config.py` | Centralized configuration: data paths, CRS constants, MOVES source type mappings, BPR parameters (alpha=0.15, beta=4 per HCM), pollutant ID-to-name mapping, free-flow speeds by road type, unit conversion factors, and plot style settings. |
| `utils.py` | Reusable functions shared across pipeline scripts: road network loading (`load_road_network`), camera API access (`load_camera_data`), Voronoi polygon generation (`build_voronoi_polygons`), BPR speed-flow function, density-speed fundamental diagram, signal phase generation (`generate_signal_phase`), vehicle count data parsing (`split_data_yolo`, `split_data_own`, `split_data_by_type`), line direction calculation, and emission file reading. |

---

## Data Accessibility
| Module | Data Required | Publicly Runnable? | Description |
|:--|:--|:--:|:--|
| **1-visual** | Public webcams or videos | Yes | Fully runnable with open imagery (e.g., [NYC DOT](https://webcams.nyctmc.org/api/cameras/)). |
| **2-dta** | OpenStreetMap network + mobile-phone OD data | Partial | Network setup (from OpenStreetMap) and [DTALite simulation](https://github.com/asu-trans-ai-lab/DTALite/tree/main) are open; OD calibration from proprietary mobility data (e.g., Cuebiq, SafeGraph, NY MPO) is restricted; fundamental diagram calibration from proprietary traffic flow data (e.g., INRIX, TomTom) is restricted. |
| **3-moves** | DTALite outputs + MOVES-Matrix engine | Partial | Input-generation scripts are open. [MOVES-Matrix](https://tse.ce.gatech.edu/development-of-moves-matrix/) and county-level emission-factor matrices must be obtained separately from Georgia Tech. |
| **4-scenario** | MOVES-Matrix output tables | Partial | Scenario evaluation and visualization run fully with the outputs of MOVES-Matrix. |

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| BPR alpha / beta | 0.15 / 4 | Highway Capacity Manual (HCM) |
| Free-flow speed (motorway / primary / secondary / residential) | 70 / 60 / 40 / 30 mph | Calibrated from INRIX speed data |
| Signal green ratio (default) | 0.5 | Symmetric split assumption |
| Vehicle deceleration | 0.2g (4.385 mph/s) | Standard comfortable deceleration |
| MOVES source types | 13 types (ID 11-62) | EPA MOVES model specification |
| Target pollutants | CO, NOx, CO2, PM2.5 | — |
| Unit conversion (meter to mile) | 0.000621371 | — |

---

## Scenario Evaluations

The framework supports evaluating several realistic interventions:

| Scenario Code | Description |
|---------------|-------------|
| `s_raw` | Baseline observed travel behavior |
| `s_mode10/20/30` | Mode shift to public transit (10-30%) |
| `s_peak10/20/30` | Departure time shift from peak hours (10-30%) |
| `s_cong_2/4/6/8` | Weeks after NYC congestion pricing launch |
| `ns`, `nsp`, `nvo`, `nv` | Ablation analysis (w/o signal control, average speed, average volume, average fleet composition) |
| `cd`, `te`, `hf`, `ss` | Real-world disruptions (COVID-19, Thanksgiving, Henri flooding, snowstorm) |

---

## Quick-Start

**Python >= 3.10**

1. Update paths in `config.py` for your local environment.
2. Install dependencies:
   ```bash
   pip install pandas geopandas numpy matplotlib seaborn tqdm shapely scipy \
               timm torch fastai ultralytics contextily mapclassify requests imageio
   ```
3. Install external tools:
   - [DTALite](https://github.com/asu-trans-ai-lab/DTALite) for dynamic traffic simulation
   - [MOVES-Matrix](https://tse.ce.gatech.edu/development-of-moves-matrix/) for emission modeling
4. Run the pipeline sequentially:
   ```bash
   python 1-visual/0.0_image2video.py
   python 1-visual/0.1_binglabeling.py
   python 1-visual/0.2_vehicle_classification.py
   python 1-visual/1.0_camera2traffic.py
   python 1-visual/1.1_camera_traffic_analysis.py
   python 1-visual/1.2_camera_signal_analysis.py
   python 2-dta/2.0_read_demand.py
   python 2-dta/2.1_dtalite_run.py
   python 2-dta/2.2_dtalite_analysis.py
   python 3-moves/3.0_moves_prepare_link.py
   python 3-moves/3.1_moves_matrix.py
   python 3-moves/3.2_emission_results.py
   python 4-scenario/4.0_scenario_moves_prepare_link.py
   python 4-scenario/4.1_scenario_results.py
   ```

---