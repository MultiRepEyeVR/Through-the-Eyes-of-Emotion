# Through the Eyes of Emotion

This repository contains the dataset and code implementation for the paper **Through the Eyes of Emotion: A Multi-faceted Eye Tracking Dataset for Emotion Recognition in Virtual Reality** by **Anonymous Author(s)**. It includes:
- **Multi-faceted Eye Tracking Dataset for Emotion Recognition**, featuring periocular videos, high-frequency eye-tracking signals, VR scene recordings, and IMU data.
- **Data Collection and Annotation Tools**, implemented in Unity and Label Studio.
- **Emotion Recognition Model and Benchmarks**, demonstrating the benefits of fusing periocular and eye movement signals.

For questions on this repository, please contact the **Annoynmous Authors**.

## Outline
- [Contributions](#contributions)
- [Data Collection](#data-collection)
- [Dataset Download](#dataset-download)
- [Data Collection and Annotation Tools](#data-collection-and-annotation-tools)
- [Results](#results)
- [Code](#code)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Contributions
Our work advances VR-based emotion recognition in three key ways:
1. **Multi-faceted Eye Tracking**  
   - **Periocular videos** captured at 120 fps (400 × 400) using near-eye cameras, enabling analysis of micro-expressions and eyebrow movements.  
   - **Gaze** recorded at 240 Hz and **pupil diameter** recorded at 120 Hz.  
2. **Discrete Emotion Annotations**  
   - Emotion labels based on Ekman's seven basic emotions (Happiness, Sadness, Fear, Disgust, Anger, Surprise, Neutral).  
   - Segment-based intensity ratings (1–10) provided by participants to capture emotion dynamics.  
3. **Open-source Collection Toolchain**  
   - Unity-based VR environment with a curved display and mid-peripheral field of view for effective 2D video stimuli.  
   - Data Recording UI for synchronized capture of VR scene, periocular video, eye-tracker, and IMU signals.  
   - Label Studio interface for segment annotation and intensity labeling.


## Data Collection

### 1. Stimuli Selection

The video stimuli used in our data collection were selected in two stages, each involving a separate set of 14 short video clips designed to evoke the seven basic emotions (Surprise, Happiness, Anger, Fear, Disgust, Sadness, and Neutral), with two clips per emotion. The detailed information for each set is provided in the paper (see Tables 9 and 10 in the Appendix).

All video clips were chosen from the validated database by Zupan et al. ([*Eliciting emotion ratings for a set of film clips: A preliminary archive for research in emotion*](https://www.tandfonline.com/doi/full/10.1080/00224545.2020.1758016)), based on emotional intensity ratings collected from 113 participants. Emotional intensity was rated on a scale from 1 (not at all) to 9 (extremely).

For each emotion, we selected the clips with the highest average emotional intensity, prioritizing both strong emotional impact and diversity in content and scenarios. Each video clip was accompanied by a brief content sentence, presented to participants before playback to provide context and aid in emotional engagement.

This selection process ensured robust and generalizable emotional elicitation across a wide range of narrative situations.

The stimuli in **Study One** includes:
| Emotion   | Film (Video Clip)             | Year | Start Time | Total Time |
|-----------|-------------------------------|------|------------|------------|
| Anger     | 12 Years A Slave              | 2013 | 30:02      | 1:55       |
| Anger     | Enough                        | 2002 | 18:04      | 1:59       |
| Disgust   | American History X            | 1998 | 54:00      | 2:29       |
| Disgust   | Limitless (blood)             | 2011 | 1:27:35    | 0:46       |
| Fear      | Kings of Summer               | 2013 | 1:15:56    | 1:01       |
| Fear      | The Conjuring                 | 2013 | 38:27      | 2:21       |
| Happiness | Soul Surfer (homeless girl)   | 2011 | 1:16:18    | 2:31       |
| Happiness | Lottery Ticket                | 2010 | 25:00      | 1:51       |
| Neutral   | Ex Machina                    | 2014 | 1:04:14    | 0:43       |
| Neutral   | Rudderless (business meeting) | 2014 | 2:28       | 0:30       |
| Sadness   | My Sister's Keeper  | 2009 | 38:50      | 1:46       |
| Sadness   | Still Alice                   | 2014 | 45:00      | 2:22       |
| Surprise  | One Day                       | 2011 | 1:26:06    | 0:38       |
| Surprise  | Deep Blue Sea                 | 1999 | 58:47      | 1:34       |

The stimuli in **Study Two** includes:
| Emotion   | Film (Video Clip)             | Year | Start Time | Total Time |
|-----------|-------------------------------|------|------------|------------|
| Anger     | Crash                         | 2004 | 15:50      | 1:32       |
| Anger     | The Hunting Ground            | 2016 | 10:15      | 2:02       |
| Disgust   | Wild                          | 2014 | 1:15       | 1:02       |
| Disgust   | Slumdog Millionaire (blinded) | 2008 | 30:33      | 1:13       |
| Fear      | The Life Before Her Eyes      | 2007 | 5:08       | 2:00       |
| Fear      | Insidious                     | 2010 | 44:30      | 2:31       |
| Happiness | Forrest Gump (reunion)        | 1994 | 1:06:54    | 0:51       |
| Happiness | Soul Surfer (surfing)         | 2011 | 50:05      | 1:49       |
| Neutral   | Good Will Hunting             | 1997 | 1:01:24    | 1:28       |
| Neutral   | The Other Woman               | 2014 | 4:49       | 0:45       |
| Sadness   | I Am Sam                      | 2001 | 1:32:45    | 1:48       |
| Sadness   | My Sister's Keeper (doctor)   | 2009 | 38:50      | 1:40       |
| Surprise  | The Call (parking garage)     | 2013 | 22:22      | 0:26       |
| Surprise  | Joe                           | 2013 | 26:41      | 0:32       |


### 2. Data collection setup
The dataset is collected from 20 subjects, with demographics of the subjects are as below:
| Characteristic | Details |
|---------------------|-------------------------------------------------------------------|
| Total Number | 26 Participants |
| Gender Distribution | Female (11), Male (15) |
| Age Range | Min: 20, Max: 41, Mean: 26.2 ± 4.2 |
| Ethnicity | Caucasian (11), East Asian (7), Middle Eastern (2), South Asian (4), African (1), Southeast Asian (1) |


 The data was collected using add-on eye tracking on HTC VIVE Pro from [Pupil Labs](https://pupil-labs.com/products/vr-ar). During the data collection, the subejcts wear and VR headset and sits on a chair while free to move any part of their body. Note that calibration is only performed once for each subject before the presenting the stimuli. We also gathered users' self-reported emotion ratings for each stimuli, ranging on the scale of 0 to 10, where 0 is the least intensity and 10 is the most intensity. Threshold 6 is taken indicated the subject is experiencing the emotion.
Example of data collected setup:

<p align="center">
    <img src="./figures/data_collection_setup.jpg" alt="Data Collection" width="600" height="auto">
</p>
Figure: (a) Pupil Labs add-on eye tracking on HTC VIVE Pro; (b) example of a subject during the data collection; (c) field view of the subject, captured by HTC VIVE Pro.


### 3. Data collection procedure
Each of the 14 stimuli is presented to each subject once. The order of the stimuli is constant for each subject, and it follows the order of low intensity to high intensity. After each stimuli, the subject is asked to rate their emotion intensity on different segments of the stimuli, which they could freely choose which segment to place and to label.

The periocular recordings are recorded at 120fps, with resolution of 400x400. The pupil diameter measurements are recorded at 120Hz, and the gaze estimation is calculated at 240Hz.

The videos below show the procedure of the data collection, with gaze estimation overlaying on the stimuli, and the pupil diameter measurements overlaying on the periocular recordings.

| Emotion  | Field of View Example | Left Eye Recording | Right Eye Recording |
|----------|----------|---------|---------|
| Disgust  | <img src="./figures/disgust/world (1).gif" alt="Disgust Field of View" width="250" height="auto"> | <img src="./figures/disgust/eye0 (1).gif" alt="Disgust Left Eye" width="125" height="auto"> | <img src="./figures/disgust/eye1 (1).gif" alt="Disgust Right Eye" width="125" height="auto"> |
| Fear     | <img src="./figures/fear/World video.gif" alt="Fear Field of View" width="250" height="auto"> | <img src="./figures/fear/Eye video.gif" alt="Fear Left Eye" width="125" height="auto"> | <img src="./figures/fear/Eye 1.gif" alt="Fear Right Eye" width="125" height="auto"> |
| Surprise | <img src="./figures/surprise/world (3).gif" alt="Surprise Field of View" width="250" height="auto"> | <img src="./figures/surprise/eye0 (3).gif" alt="Surprise Left Eye" width="125" height="auto"> | <img src="./figures/surprise/eye1 (3).gif" alt="Surprise Right Eye" width="125" height="auto"> |
| Anger    | <img src="./figures/anger/world (2).gif" alt="Anger Field of View" width="250" height="auto"> | <img src="./figures/anger/eye0 (2).gif" alt="Anger Left Eye" width="125" height="auto"> | <img src="./figures/anger/eye1 (2).gif" alt="Anger Right Eye" width="125" height="auto"> |

Each GIF shows the subject's field of view in real-time, with gaze estimation overlaying on the stimuli, and the pupil diameter measurements overlaying on the periocular recordings.


## Dataset Download

### Preview Dataset (For Review)
A sample dataset containing one complete subject's data is available for preview during the paper review process. The dataset includes:
- Raw periocular recordings from both eyes (120 FPS, 400x400 resolution)
- Gaze estimation data (~240 Hz)
- Pupil diameter measurements (~120 Hz)
- Voice recordings
- IMU data from the VR headset
- Scene recordings (subject's field of view)
- Self-reported emotion ratings for all 14 stimuli

Download the sample dataset from [Zenodo](https://zenodo.org/records/14165275).

### Complete Dataset
The complete dataset containing all 26 subjects' data will be made publicly available upon paper acceptance.

**Soon available**

### Dataset Structure
The dataset follows a hierarchical organization designed for efficient access and processing:
```
dataset/
├── P01/
│ ├── 0a/
│ ├── 0b/
│ ├── 1a/
│ ├── 1b/
│ ├── 2a/
│ ├── 2b/
│ ├── 3a/
│ ├── 3b/
│ ├── 4a/
│ ├── 4b/
│ ├── 5a/
│ ├── 5b/
│ ├── 6a/
│ └── 6b/
├── P02/
│ └── [same structure as P01]
...
├── P20/
│ └── [same structure as P01]
├── P21/
│ ├── 0c/
│ ├── 0d/
│ ├── 1c/
│ ├── 1d/
│ ├── 2c/
│ ├── 2d/
│ ├── 3c/
│ ├── 3d/
│ ├── 4c/
│ ├── 4d/
│ ├── 5c/
│ ├── 5d/
│ ├── 6c/
│ └── 6d/
├── P22/
│ └── [same structure as P21]
...
└── P26/
└── [same structure as P21]
```
- **P01–P20**: Subjects from Study One. Each has session folders labeled `0a/0b`, `1a/1b`, ..., `6a/6b` (where `a` and `b` indicate the two clips per emotion from Study One).
- **P21–P26**: Subjects from Study Two. Each has session folders labeled `0c/0d`, `1c/1d`, ..., `6c/6d` (where `c` and `d` indicate the two clips per emotion from Study Two).

Each session folder (e.g., `0a/`, `1d/`) contains:
- `audio/` (e.g., `micRec.wav`)
- `gaze_pupil/` (`gaze.csv`, `pupil.csv`)
- `imu/` (`imu.csv`)
- `periocular/` (`eye0.mp4`, `eye1.mp4`)
- `scene/` (`world.mp4`)

Additionally, we provide a `user_label.json` file containing crucial timing information about emotion elicitation for each participant. This file maps the specific timestamps (in MM:SS format) when each participant reported experiencing the intended emotion during different stimuli sessions.

**Session coding:**
- For P01–P20:  
  - `0a/0b`: Neutral (Study One)  
  - `1a/1b`: Surprise (Study One)  
  - `2a/2b`: Happiness (Study One)  
  - `3a/3b`: Sadness (Study One)  
  - `4a/4b`: Anger (Study One)  
  - `5a/5b`: Disgust (Study One)  
  - `6a/6b`: Fear (Study One)
- For P21–P26:  
  - `0c/0d`: Neutral (Study Two)  
  - `1c/1d`: Surprise (Study Two)  
  - `2c/2d`: Happiness (Study Two)  
  - `3c/3d`: Sadness (Study Two)  
  - `4c/4d`: Anger (Study Two)  
  - `5c/5d`: Disgust (Study Two)  
  - `6c/6d`: Fear (Study Two)

## Data Collection and Annotation Tools

This project utilizes two primary tools for data collection and annotation:

### 1. Data Recording Interface

A custom Unity-based user interface has been developed for data recording. To access this tool:

1. Download the Unity project from [Zenodo](https://doi.org/10.5281/zenodo.14066626).
2. Open the project in the Unity Editor.

The interface is designed as follows:
<p align="center">
    <img src="./figures/Emotion Recognition UI.jpg" alt="Data Recording Interface" width="800" height="auto">
</p>

### 2. Data Annotation Interface

For data annotation, we employ a modified version of [Label Studio](https://labelstud.io/). Due to potential compatibility issues with the latest release, we recommend using our adapted version:

1. Download our modified Label Studio version from [soon available]().
2. Follow the installation instructions provided in the download.

The annotation interface is illustrated below:

<p align="center">
    <img src="./figures/Emotion Recognition UI (1).jpg" alt="Data Annotation Interface" width="800" height="auto">
</p>

<!-- #### Configuration for Large Datasets

To accommodate large datasets, adjust the following parameters in `label-studio/label_studio/core/settings/base.py`:

```python
DATA_UPLOAD_MAX_MEMORY_SIZE = int(get_env('DATA_UPLOAD_MAX_MEMORY_SIZE', 25 * 1024 * 1024 * 1024))
DATA_UPLOAD_MAX_NUMBER_FILES = int(get_env('DATA_UPLOAD_MAX_NUMBER_FILES', 10000))
TASKS_MAX_NUMBER = 1000000000
``` -->

#### Deployment Using Docker

To deploy the modified Label Studio:

- Build & Run with Docker Compose:
   ```
   docker-compose up
   ```

#### Access the annotation interface
Access the annotation interface at `http://localhost:8080` after deployment.


## Results

We compare three variants throughout:

* **Eye-movement only** – gaze + pupil-diameter time–series  
* **Periocular video only** – periocular recordings of each eye  
* **Multi-faceted (ours)** – fusion of periocular and eye-movement streams  

> **Note:** Results 1-3 are based on **the 20 participants and 14 video clips of *Study One***.  
> Result 4 evaluates **cross-study transfer from Study One → Study Two** (6 new participants and 14 new clips).

---

### 1 · Subject-independent pre-training    
*Five-fold CV on Study One: train on 16 subjects (two sessions each), test on the 4 held-out subjects.*  

| Window (s) | 0.5 | 1.0 | 1.5 | 2.0 |
|------------|-----|-----|-----|-----|
| Eye-mov. **F1 / Acc** | 0.26 / 0.28 | 0.27 / 0.29 | 0.29 / 0.31 | 0.30 / 0.31 |
| Periocular **F1 / Acc** | **0.45 / 0.45** | 0.44 / 0.44 | 0.46 / 0.47 | 0.45 / 0.45 |
| **Multi-faceted F1 / Acc** | 0.43 / 0.43 | **0.44 / 0.44** | **0.46 / 0.47** | **0.52 / 0.52** |

With a 2 s window the fusion model raises F1 by **+73.3%** over eye movements alone and **+15.6%** over periocular video.   

---

### 2 · 10 % fine-tuning (subject-dependent)    
*The pre-trained network is fine-tuned on the **first 10 %** of each recording, then evaluated either on the same session or on the subject's other session (unseen clips).*  

| Setting | Eye-mov. | Periocular | **Multi-faceted** |
|---------|----------|-----------|-------------------|
| Same-session **F1 / Acc** | 0.46 / 0.49 | 0.82 / 0.82 | **0.84 / 0.85** |
| Cross-session **F1 / Acc** | 0.23 / 0.25 | 0.54 / 0.55 | **0.70 / 0.71** |

Fusion adapts best—especially when testing on **new video stimuli**.

---

### 3 · Few-shot cross-session adaptation    
*The model is fine-tuned on **k labelled segments per class** from Session A of each subject and tested on Session B (different clips).*  

| Shots / class | 1 | 2 | 3 | 4 | 5 |
|---------------|---|---|---|---|---|
| Periocular **F1 / Acc** | 0.53 / 0.53 | 0.56 / 0.56 | 0.57 / 0.57 | 0.57 / 0.57 | 0.57 / 0.57 |
| **Multi-faceted F1 / Acc** | **0.67 / 0.66** | **0.68 / 0.67** | **0.69 / 0.69** | **0.70 / 0.69** | **0.70 / 0.70** |

With just **one** segment per emotion, fusion is already outperforming the periocular baseline; we omit eye-movement results here due to its very low performance. 

---

### 4 · Few-shot cross-**study** transfer  (Study One → Study Two)  
*Pre-train on all of Study One, then fine-tune and test **subject-specifically** on Study Two (new users **and** new clips).*  

| Shots / class | 1 | 2 | 3 | 4 | 5 |
|---------------|---|---|---|---|---|
| Eye-mov. **F1 / Acc** | 0.12 / 0.16 | 0.14 / 0.17 | 0.14 / 0.17 | 0.14 / 0.17 | 0.15 / 0.18 |
| Periocular **F1 / Acc** | 0.52 / 0.52 | 0.52 / 0.52 | 0.53 / 0.53 | 0.54 / 0.54 | 0.55 / 0.55 |
| **Multi-faceted F1 / Acc** | **0.57 / 0.58** | **0.59 / 0.60** | **0.59 / 0.60** | **0.61 / 0.61** | **0.62 / 0.62** |

Fusion more than **quadruples** the 1-shot F1 of the eye-movement baseline and maintains an advantage over periocular up to 5 shots, evidencing robust generalisation to **unseen subjects *and* unseen stimuli**.

---

#### Key take-aways  

* Fusion of periocular recordings with eye-movements **wins across every scenario**.  
* It is highly **data-efficient**, needing as little as one labelled segment per emotion to surpass stronger single-stream baselines.  
* Longer eye-movement windows help that baseline, but fusion still leads by ≥ 10 F1 points in Study One and by ≥ 42 F1 points in cross-study transfer.

These findings confirm that fine-grained periocular cues and high-frequency eye dynamics form a complementary, powerful signal for emotion recognition in VR.


## Code
This repository provides comprehensive code for data preprocessing, model implementation, and training:

1. **Data Preprocessing Pipeline**: 
   - Recommended Complete pipeline for converting raw data into training-ready datasets (`process_dataset.ipynb`)
   - Feature extraction tools for:
     - Periocular recordings
     - Pupil measurements
     - Gaze coordinates
     - Combined modality processing
   
The processed data follows this structure:
```
processed_dataset/
├── [processing_parameters]/      # Directory named with processing configuration
    ├── index_info.json           # Dataset metadata and processing parameters
    ├── P01.h5                    # Processed data for participant 1
    ├── P02.h5                    # Processed data for participant 2
    └── ...                       # Additional participant files
```
The `index_info.json` file contains metadata for each participant's processed data, structured for efficient data loading during training:
```
{
    "P01": {
        // Number of processed samples per emotion session
        "0a_amount": 43,         # Neutral session a
        "0b_amount": 31,         # Neutral session b
        "1a_amount": 5,          # Surprise session a
        "1b_amount": 10,         # Surprise session b
        // ... (similar for other emotion sessions)
        
        "min_index": 0,          # Starting index in the dataset
        "max_index": 869         # Ending index in the dataset
    },
    // ... entries for other participants
}
```

Each participant's H5 file contains:
```
subject.h5/
├── labels/                      # Shape: (N,)
│   └── Compound dtype:
│       ├── subject              # (int64) Subject number
│       ├── label                # (int64) Emotion label
│       └── session_label        # (string) Session identifier
│
├── gaze_pupil/                  # Shape: (N, gaze_pupil_features)
│   └── Float32 array of gaze coordinates and pupil diameter
│
├── eye0/                        # Shape: (N, H, W, C)
│   └── Float32 array of right eye frames (normalized 0-1)
│
└── eye1/                        # Shape: (N, H, W, C)
    └── Float32 array of left eye frames (normalized 0-1)
```

2. **Training Framework**:
   - Complete training pipeline with configuration files (`train_model.ipynb`)
   - Data loading and augmentation utilities
   - Experiment tracking and logging

The code is organized to help researchers reproduce our results and build upon our work. Model weights and will be released alongside the complete dataset upon paper acceptance.

## Citation
**Soon available**

## Acknowledgement
**Soon available**
