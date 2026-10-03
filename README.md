```markdown
# Eye-Tracking & Sustained Attention Analysis for NDD Children during Gameplay

This repository contains Python workflows and scripts to process, clean, cluster, and analyze raw eye-tracker data (gaze coordinates, fixations, saccades) collected from neurodevelopmentally diverse (NDD) children engaged in interactive training tasks. 

By mapping gaze coordinates to specific game Areas of Interest (AOIs) and correlating them with in-game stimulus triggers (events like targets appearing/disappearing), this pipeline calculates key physiological indicators such as **sustained attention levels, engagement rates, cognitive load, and reaction times**.

---

## 🚀 Project Overview

Eye-tracking offers objective, non-invasive metrics to understand how children with neurodevelopmental conditions process information, manage cognitive load, and maintain focus. This toolset enables:
* **Pre-processing Raw Coordinate Streams:** Standardizing coordinate systems, filtering blink/out-of-bound errors, and aligning timelines.
* **K-Means Spatial Clustering:** Automatically discovering prominent spatial hotspots (clusters of focus) on gameplay screen backdrops.
* **Quadrant & AOI Mapping:** Evaluating whether gaze patterns are inside targeted Areas of Interest (e.g., target/distractor objects) vs. background quadrants.
* **Cognitive Load Indicators:** Analyzing fixation durations across different activities and difficulty levels.
* **Attention & Engagement Quantification:** Tracking metrics like absolute engagement (`Engagement` = Yes/No) and hit-rate dynamics (`ET_Hit` durations) relative to active target/distractor lifecycles.

---

## 📁 Directory Structure & Key Files

* **`cleaned_LVL1_data_300rows.csv` / `filtered_student_data_300rows.csv`**: Sample clean and filtered outputs mapping student fixation profiles.
* **`is_in_q3_q4.csv`**: Automated tracking of vertical visual focus across lower-screen game quadrants.
* **`attentionData.csv`**: Integrated dataset mapping temporal gaze records, object states, and dynamic attention engagement status.

---

## 🛠️ Main Components & Core Scripts

### 1. Spatial Gaze Clustering (`KMeans`)
Identifies high-density gaze points using spatial clustering on the coordinate streams:
- Groups eye coordinates into discrete clusters (e.g., $C_1, C_2, C_3$).
- Generates convex hulls surrounding cluster bounds overlayed directly onto gameplay UI screens.
- Highlights visual hot-spots dynamically per session.

### 2. AOI Validation & Quadrant Distribution
Divides the gameplay interface ($1920 \times 1080$) into 4 active quadrants ($Q_1$ to $Q_4$):
- Calculates exact coordinates falling inside user-defined interactive boundaries (yellow bounding boxes).
- Flags focus alignment to detect drift away from gameplay mechanics.

### 3. Cognitive Load Assessment
Uses fixation durations ($ms$) and spatial dispersion profiles as proxies for mental workload:
- Identifies the longest fixation durations for each subject and flags potential points of visual stalling or high cognitive friction.
- Highlights individual engagement peaks through customized density KDE heatmaps.

### 4. Sustained Attention & Distractor Engagement
Correlates timestamped game engine events (e.g., `Appear_Target_Mushroom`, `Appear_Distractor_BlueFlowers`) with immediate gaze alignment within the target's bounding box:
- **ET_Hit**: Calculated as the exact time delta from stimulus emergence to the point when the child successfully redirects attention inside the designated AOI.
- Tracks fixation counts, distractor engagement rates, and eye-tracker drop-off rates (`ET_Off`).

---

## 📦 Quick Start & Requirements

### Requirements
Ensure you have the following packages installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly scipy
```

### Usage Example
To run the primary pipeline analyzing sustained attention and generating the output engagement profiles:
```python
from your_module import sustained_attention

# Configure paths to your eye-tracker dataset and screen layout image
data_path = 'path/to/raw_gaze_data.csv'
bg_image_path = 'path/to/gameplay_screen.png'
output_path = 'path/to/attention_output.csv'

# Execute attention mapping pipeline
sustained_attention(data_path, bg_image_path, output_path)
```
```
