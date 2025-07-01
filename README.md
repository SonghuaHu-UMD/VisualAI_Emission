# Ubiquitous Data-Driven Framework for Traffic Emission Estimation and Policy Evaluation

This repository provides a full-stack urban transportation emissions analysis framework that integrates:
- Public camera feeds
- Deep learning vehicle detection/classification
- Simulation-based OD assignment
- EPA MOVES Matrix emission modeling
- Scenario testing for transportation policies (mode shift, peak spreading, congestion pricing)

The framework is built on open datasets (NYC cameras, mobile OD, OpenStreetMap) and supports large-scale experiments.

---

## 🧭 Pipeline Overview

### Phase 0: Camera & Classifier Preprocessing
```
├── 0.0_image2video.py              # Convert DOT image feeds to video
├── 0.1_binglabeling.py            # Collect training images via Bing
├── 0.2_vehicle_classification.py  # Train deep vehicle type classifiers (EfficientNet/ViT)
```

### Phase 1: Video Analytics
```
├── 1.0_camera2traffic.py          # Run detection, tracking, and classification
├── 1.1_camera_traffic_analysis.py # Analyze vehicle volumes across space and time
├── 1.2_camera_signal_analysis.py  # Extract traffic signal cycles from visual input
├── 1.3_camera_vehicle_analysis.py # Class distribution, confusion matrix, accuracy evaluation
```

### Phase 2: Traffic Simulation (DTALite)
```
├── 2.0_read_demand.py             # Generate OD demand from mobile device data
├── 2.1_dtalite_run.py             # Create simulation input files and run DTALite
├── 2.2_dtalite_analysis.py        # Parse simulation output (link-level speed and volume)
├── 2.3_VSD_analysis.py            # Fit fundamental diagrams (volume-speed-density)
```

### Phase 3: Emission Modeling (MOVES Matrix)
```
├── 3.0_moves_prepare_link.py      # Prepare MOVES inputs (link speeds, fleet mix)
├── 3.1_moves_matrix.py            # Run MOVES-Matrix batch mode
├── 3.2_emission_results.py        # Analyze emissions, spatial maps, time variation
```

### Phase 4: Policy Scenario Modeling
```
├── 4.0_scenario_moves_prepare_link.py  # Create hypothetical policy inputs (mode shift, pricing)
├── 4.1_scenario_results.py             # Evaluate impact of each scenario on emissions
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
| `cd`, `te`, `hf`  | Real-world disruptions (COVID, flooding, etc.) |

---

## 🚀 Getting Started

### 1. Install Dependencies

You’ll also need:
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [DTALite](https://github.com/asu-trans-ai-lab/DTALite)
- EPA [MOVES-Matrix](https://tse.ce.gatech.edu/development-of-moves-matrix/)

### 2. Run the Pipeline

#### A. From Video to Traffic
```bash
python 0.0_image2video.py
python 0.2_vehicle_classification.py
python 1.0_camera2traffic.py
```

#### B. Simulation & Emissions
```bash
python 2.0_read_demand.py
python 2.1_dtalite_run.py
python 3.0_moves_prepare_link.py
python 3.1_moves_matrix.py
python 3.2_emission_results.py
```

#### C. Scenario Modeling
```bash
python 4.0_scenario_moves_prepare_link.py
python 4.1_scenario_results.py
```

---

## 📊 Outputs

- **CSV & Pickle**: Link-level volume, speed, emissions (CO₂, NOx, PM, etc.)
- **PDF & PNG**: Spatial maps, boxplots, bar charts
- **Video & GIF**: Annotated traffic videos
- **GIS**: Road network shapefiles enriched with traffic/emission data

---

## 📂 Directory Structure

```bash
/NY_Image                 # Raw camera snapshots
/NY_Video                 # Converted MP4s
/Cars_Dataset             # Labeled images for classifier
/MOVES/input_*            # Inputs for different scenarios
/MOVES/output_*           # Emission output (per scenario)
/ODME_NY/                 # Simulation results
/Figures/                 # Visualizations
/Shp/                     # Road network and shapefiles
```

---

## 📚 Data Sources

- **Cameras**: NYC DOT [Webcam API](https://webcams.nyctmc.org/)
- **Road Network**: OpenStreetMap (via `osmnx` and `osm2gmns`)
- **OD Data**: Cuebiq mobile device data
- **Emissions**: EPA MOVES-Matrix (local CSV databases)


---

## 📜 License

This project is licensed under the MIT License.  
See [`LICENSE`](LICENSE) for details.
