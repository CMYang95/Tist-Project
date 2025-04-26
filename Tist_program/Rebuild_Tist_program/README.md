# TIST Project

## Overview
The TIST project is designed for video analysis and pose estimation using computer vision techniques. It leverages libraries such as OpenCV and MediaPipe to capture video, analyze frames, and visualize data related to human motion.

## Project Structure
```
Tist_program
├── app.py                  # Entry point of the application
├── src                     # Source code directory
│   ├── __init__.py        # Package initialization
│   ├── utils               # Utility functions
│   │   ├── __init__.py    # Package initialization
│   │   ├── calculations.py  # Mathematical calculations
│   │   ├── plotting.py      # Data visualization
│   │   └── tracking.py      # Object tracking functions
│   ├── config              # Configuration settings
│   │   ├── __init__.py    # Package initialization
│   │   └── settings.py     # Application settings
│   ├── video_processing     # Video processing functions
│   │   ├── __init__.py    # Package initialization
│   │   ├── capture.py      # Video capture functions
│   │   └── analysis.py     # Video analysis functions
│   └── models              # Model definitions
│       ├── __init__.py    # Package initialization
│       └── pose_estimation.py # Pose estimation functions
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:

```
python app.py
```

Make sure to have your video files ready and configured in the settings.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.