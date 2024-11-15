# Through the Eyes of Emotion

This repository contains the dataset and code implementation for the paper **Through the Eyes of Emotion: A Multi-representation Eye Tracking
Dataset for Emotion Recognition in Virtual Reality** by **Annoynmous Authors**. It contains the **Multi-representation Eye Tracking Dataset for Emotion Recognition**, code for the tool for gathering the dataset and the code for the emotion recognition model.

For questions on this repository, please contact the **Annoynmous Authors**.

## Outline
- [Data Collection](#data-collection)
- [Dataset Download](#dataset-download)
- [Data Collection and Annotation Tools](#data-collection-and-annotation-tools)
- [Code](#code)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Data Collection

### 1. Stimuli selection

The 14 stimuli are seleted from the psychology study [*Eliciting emotion ratings for a set of film clips: A preliminary archive for research in emotion*](https://www.tandfonline.com/doi/full/10.1080/00224545.2020.1758016). The 14 stimuli triggers 7 emotions which are Surprise, Happiness, Anger, Fear, Disgust, Sadness, and Neutral, as shown in the figure below.

<!-- <p align="center">
  <img src="./figures/emotion_model.png" alt="Emotion Stimuli" width="300" height="auto">
</p>
Figure 1: The Paul Ekman's six basic emotions, with neutral in addition to represent when no emotion is present. -->

<!-- The stimuli inlcudes:
| Film                                | Year | Start Time | Total Time | Emotion   |
|-------------------------------------|------|------------|------------|-----------|
| Kings of Summer                     | 2013 | 1:15:56    | 0:49       | Fear      |
| American History X                  | 1998 | 54:00      | 1:23       | Disgust   |
| Limitless (blood)                   | 2011 | 1:27:35    | 0:58       | Disgust   |
| Soul Surfer (homeless girl)         | 2011 | 1:16:18    | 1:45       | Happiness |
| Ex Machina                          | 2014 | 1:04:14    | 0:45       | Neutral   |
| Rudderless (business meeting)       | 2014 | 2:28       | 0:29       | Neutral   |
| One Day                             | 2011 | 1:26:06    | 0:25       | Surprise  |
| 12 Years A Slave                    | 2013 | 30:02      | 1:48       | Anger     |
| Enough                              | 2002 | 18:04      | 1:43       | Anger     |
| Lottery Ticket                      | 2010 | 25:00      | 1:13       | Happiness |
| My Sister's Keeper (doctor)         | 2009 | 38:50      | 1:09       | Sadness   |
| Still Alice                         | 2014 | 45:00      | 1:26       | Sadness   |
| Deep Blue Sea                       | 1999 | 58:47      | 1:57       | Surprise  |
| The Conjuring                       | 2013 | 38:27      | 2:06       | Fear      | -->

The stimuli includes:
| Emotion   | Film                          | Year | Start Time | Total Time |
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
| Sadness   | My Sister's Keeper (doctor)   | 2009 | 38:50      | 1:46       |
| Sadness   | Still Alice                   | 2014 | 45:00      | 2:22       |
| Surprise  | One Day                       | 2011 | 1:26:06    | 0:38       |
| Surprise  | Deep Blue Sea                 | 1999 | 58:47      | 1:34       |

### 2. Data collection setup
The dataset is collected from 20 subjects, with demographics of the subjects are as below:
| Characteristic | Details |
|---------------------|-------------------------------------------------------------------|
| Total Number | 20 Participants |
| Gender Distribution | Female (9), Male (11) |
| Age Range | Min: 20, Max: 30, Mean: 24.9 ± 2.3 |
| Ethnicity | Caucasian (11), East Asian (4), Middle Eastern (2), South Asian (1), African (1) |


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
The complete dataset containing all 20 subjects' data will be made publicly available upon paper acceptance.

**Soon available**

### Dataset Structure
The dataset follows a hierarchical organization designed for efficient access and processing:
```
dataset/
├── P01/
│   ├── 0a/
│   │   ├── audio/
│   │   │   └── micRec.wav
│   │   ├── gaze_pupil/
│   │   │   ├── gaze.csv
│   │   │   └── pupil.csv
│   │   ├── imu/
│   │   │   └── imu.csv
│   │   ├── periocular/
│   │   │   ├── eye0.mp4
│   │   │   └── eye1.mp4
│   │   └── scene/
│   │       └── world.mp4
│   ├── 0b/
│   ├── 1a/
│   ├── 1b/
│   ├── 2a/
│   ├── 2b/
│   ├── 3a/
│   ├── 3b/
│   ├── 4a/
│   ├── 4b/
│   ├── 5a/
│   ├── 5b/
│   ├── 6a/
│   └── 6b/
├── P02/
│   └── [same structure as P01]
...
└── P20/
    └── [same structure as P01]
```
Additionally, we provide `user_label.json` which contains crucial timing information about emotion elicitation for each participant. This file maps the specific timestamps (in MM:SS format) when each participant reported experiencing the intended emotion during different stimuli sessions. For example, in session "2a", some participants reported the emotion from 0:19 while others from 1:20.

The sessions are coded as follows:
- 0a/0b: Neutral stimuli
- 1a/1b: Surprise stimuli
- 2a/2b: Happiness stimuli
- 3a/3b: Sadness stimuli
- 4a/4b: Anger stimuli
- 5a/5b: Disgust stimuli
- 6a/6b: Fear stimuli

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

Each participant's H5 file contains:
```
subject.h5/
├── labels/                       # Shape: (N,)
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

2. **Model Implementation**:
   - Implementation of our proposed multi-modal emotion recognition architecture
   - Modular design supporting different backbone networks
   - Custom layers for multi-modal fusion
   - Evaluation metrics and visualization tools

3. **Training Framework**:
   - Complete training pipeline with configuration files (`train_model.ipynb`)
   - Data loading and augmentation utilities
   - Experiment tracking and logging

The code is organized to help researchers reproduce our results and build upon our work. Model weights and will be released alongside the complete dataset upon paper acceptance.

**Note**: Detailed documentation and usage instructions will be available soon.

## Citation
**Soon available**

## Acknowledgement
**Soon available**
