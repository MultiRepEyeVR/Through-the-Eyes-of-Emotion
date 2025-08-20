# Through the Eyes of Emotion

This repository contains the dataset and code implementation for the paper **Through the Eyes of Emotion: A Multi-faceted Eye Tracking Dataset for Emotion Recognition in Virtual Reality**. It includes:
- **Multi-faceted Eye Tracking Dataset for Emotion Recognition**, featuring periocular videos, high-frequency eye-tracking signals, VR scene recordings, and IMU data.
- **Data Collection and Annotation Tools**, implemented in Unity and Label Studio.

For questions on this repository, please contact Tongyun Yang (tongyunyang AT outlook.com) or Guohao Lan (g.lan AT tudelft.nl).

## Outline
- [Contributions](#contributions)
- [Data Collection](#data-collection)
- [Dataset Download](#dataset-download)
- [Data Collection and Annotation Tools](#data-collection-and-annotation-tools)
- [Implementation for Evaluation](#code)
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

The stimuli used in **Study One** includes:
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

The stimuli used in **Study Two** includes:
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
The dataset is collected from 26 subjects, with demographics of the subjects are as below:
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

To comply with single-file size limits on the hosting platform, both the raw and processed datasets are uploaded as multi‑part archives. Please download all parts for each dataset and then combine them locally before extracting.

- Raw dataset (parts): [Zenodo-1](https://zenodo.org/records/16794721), [Zenodo-2](https://zenodo.org/records/16794737), [Zenodo-3](https://zenodo.org/records/16794742)
- Processed dataset (parts): [Zenodo-1](https://zenodo.org/records/16794776), [Zenodo-2](https://zenodo.org/records/16794794), [Zenodo-3](https://zenodo.org/records/16794809)
- Processed dataset (full): [Zenodo](https://zenodo.org/records/16790658)

Combine the parts and extract:
```bash
# Join parts into a single zip
cat <name>.zip.part-* > <name>.zip

# Unzip
unzip <name>.zip
```

Note: Ensure all parts (e.g., .part-01, .part-02, .part-03) are fully downloaded for each dataset before running the join command.

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

1. Download the Unity project with corresponding scripts from [Zenodo](https://doi.org/10.5281/zenodo.16778708).
2. Open the project in the Unity Editor.

The interface is designed as follows:
<p align="center">
    <img src="./figures/Emotion Recognition UI.jpg" alt="Data Recording Interface" width="800" height="auto">
</p>

**Important Setup Note**: When opening the project in Visual Studio (`.sln` file), you will find a hardcoded path in `PythonScriptsManagerScript.cs` that points to a local Python script. You need to:
1. Place the required Python scripts (the directory `Scripts`) including `prepareLabelStudio.py` and `launch_pupil.py` (downloaded in addition to the Unity project) in your desired location
2. Update the absolute paths in `PythonScriptsManagerScript.cs` to point to your Python script location
3. Specify the absolute path for data storage location on line`76` (by default, we recommend `Desktop/ER_in_VR_Recordings/`)

### 2. Data Annotation Interface

For data annotation, we employ a modified version of [Label Studio](https://labelstud.io/). Our custom implementation is based on a specific version of Label studio with some special designs for the emotion annotation tasks.

#### Installation Options

1. **Official Label Studio**: Download from [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio/)
2. **Our Modified Version**: Download our customized version optimized for emotion recognition tasks from [Zenodo](https://doi.org/10.5281/zenodo.16778800).

Both versions are expected to be compatible with our annotation workflow.

The annotation interface is illustrated below:

<p align="center">
    <img src="./figures/Emotion Recognition UI (1).jpg" alt="Data Annotation Interface" width="800" height="auto">
</p>

### 3. Usage

#### Configuration Setup

To handle our large-scale dataset, modify the following parameters in `label-studio/label_studio/core/settings/base.py`:

```python
DATA_UPLOAD_MAX_MEMORY_SIZE = int(get_env('DATA_UPLOAD_MAX_MEMORY_SIZE', 25 * 1024 * 1024 * 1024))
DATA_UPLOAD_MAX_NUMBER_FILES = int(get_env('DATA_UPLOAD_MAX_NUMBER_FILES', 10000))
TASKS_MAX_NUMBER = 1000000000
```

For authentication, add your Label Studio identification token in the python file mentioned above `prepareLabelStudio.py`:
```python
API_KEY = 'YOUR SECRET KEY HERE'
# example: API_KEY = 'c6270738f719fjk18904ujf1ehhf729'
```

#### Deployment

To deploy the modified Label Studio:

- Build & Run with Docker Compose:
   ```
   docker-compose up
   ```

#### Access the annotation interface
Access the annotation interface at `http://localhost:8080` after deployment.

## Implementation for Evaluation
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

The code is organized to help researchers reproduce our results and build upon our work. The complete dataset will be released upon paper acceptance.

## Citation

Please cite the paper in your publications if it helps your research.
```
@article{yang2025VREyeEmotion,
  title={Through the Eyes of Emotion: A Multi-faceted Eye Tracking Dataset for Emotion Recognition in Virtual Reality},
  author={Yang, Tongyun and Regmi, Bishwas and Du, Lingyu and and Bulling, Andreas and Zhang, Xucong and Lan, Guohao},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={9},
  number={3},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```

## Acknowledgement
This work was supported in part by the Meta Research Award, SURF Research Cloud grant EINF-6360, and the EU’s Horizon Europe HarmonicAI project under the HORIZON MSCA-2022-SE-01 scheme with grant agreement number 101131117. The contents of this paper do not necessarily reflect the positions or policies of the funding agencies.
