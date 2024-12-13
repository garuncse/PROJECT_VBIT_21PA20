# How to Run the Intrusion Detection System App

## Overview

python app.py --port 5001

## Prerequisites

- Python 3.10
- Git
- Virtual Environment (Optional)
- Conda (Optional)

## Setup Instructions



### 1. Create a Virtual Environment (Optional)

#### Using Virtual Environment

\```bash
python -m venv intrusion_detection_env
# Activate the virtual environment
# On Windows
.\intrusion_detection_env\Scripts\activate
# On macOS and Linux
source intrusion_detection_env/bin/activate
\```

#### Using Conda

\```bash
conda create --name intrusion_detection_env python=3.10
conda activate intrusion_detection_env
\```

### 3. Install Dependencies

\```bash
pip install -r requirements.txt
\```

## Running the App

### Start the Application

\```bash
python app.py
\```

## Usage

1. Open a web browser and navigate to `http://localhost:5000`.
2. Click on the "Choose File" button to select a `.csv` file containing network activity data.
3. Click on the "Upload & Analyze" button to initiate the analysis process.
4. To view detailed model information and metrics, click on the "Model Info" button.

## Troubleshooting

- Ensure that all dependencies are correctly installed.
- If encountering any issues, refer to the console or terminal output for error messages.

## Conclusion

Follow these instructions to set up and run the Intrusion Detection System application. For any further assistance or queries, please reach out for support.
