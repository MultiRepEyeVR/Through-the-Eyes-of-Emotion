# Multi-representation Emotion Recognition in Immersive Environments

This repository contains the dataset and code implementation for the paper **Multi-representation Emotion Recognition in Immersive Environments** by **Annoynmous Authors**. It contains the **Multi-representation Emotion Dataset**, code for the tool for gathering the dataset and the code for the emotion recognition model.

For questions on this repository, please contact the **Annoynmous Authors**.

## Outline
- [Data Collection](#data-collection)
- [Dataset Download](#dataset-download)
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

The stimuli inlcudes:
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
| The Conjuring                       | 2013 | 38:27      | 2:06       | Fear      |

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
Figure 1: (a) Pupil Labs add-on eye tracking on HTC VIVE Pro; (b) example of a subject during the data collection; (c) field view of the subject, captured by HTC VIVE Pro.


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
**Soon available**

## Data Collection and Annotation Tools

This project utilizes two primary tools for data collection and annotation:

### 1. Data Recording Interface

A custom Unity-based user interface has been developed for data recording. To access this tool:

1. Download the Unity project from [soon available]().
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

#### Configuration for Large Datasets

To accommodate large datasets, adjust the following parameters in `label-studio/label_studio/core/settings/base.py`:

```python
DATA_UPLOAD_MAX_MEMORY_SIZE = int(get_env('DATA_UPLOAD_MAX_MEMORY_SIZE', 25 * 1024 * 1024 * 1024))
DATA_UPLOAD_MAX_NUMBER_FILES = int(get_env('DATA_UPLOAD_MAX_NUMBER_FILES', 10000))
TASKS_MAX_NUMBER = 1000000000
```

#### Deployment Using Docker

To deploy the modified Label Studio:

1. Build the Docker image:
   ```
   docker build -t heartexlabs/label-studio:latest .
   ```

2. Run the Docker container:
   ```
   docker run -it -p 8080:8080 heartexlabs/label-studio:latest
   ```

Access the annotation interface at `http://localhost:8080` after deployment.



## Code
**Soon available**

## Citation
**Soon available**

## Acknowledgement
**Soon available**
