# Face Enhancer

## Description
The Face Enhancer improves the quality and details of facial features in images.

# Caution

The face enhancer calls the face feature analyser which itself calls the face detector.
**Of note**: our implementation of processors calling each other has **NOT** been defined at the moment.
This may trigger code redundancies and maintenance difficulty accross the different parts of facefusion !

For now, I circumvent this issue by cloning the face feature analyzer repo from github and running it on marie image.
I then used the resulting JSON file from the face feature analyzer to test this processor.

## Guidelines

### Input and Output

- **Input**: Image file (e.g., `.jpg`)
- **Output**: JSON file containing detailed face data

### Pipeline Structure

The `face_analyser_event` follows a structured pattern: `preprocess` -> `inference` -> `postprocess`. These three services are isolated and do not call each other directly. Instead, they are orchestrated by the `Pipeline` class.

### Service Isolation

Each service (preprocessing, inference, postprocessing) is designed to be testable and can have its own security and monitoring rules. This modular approach ensures that each component can be independently verified and maintained.

### Testing

To ensure the reliability of each service, comprehensive tests are provided. Each service can be tested individually, and the entire pipeline can be validated using the provided test scripts.

# Dependencies

This face analyzer requires minimal dependencies:

- Libraries: `numpy`, `opencv-python`, and `onnxruntime`
- Models: `gfpgan_1.4.onnx`, `dfl_xseg.onnx`

# Installation

1. Create a Python 3.10 virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and place the necessary models in a dedicated repo:

   - Face Enhancer: [gfpgan_1.4.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx)
   - Face Occluder: [dfl_xseg.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.onnx)

# Running the Pipeline
```python
import cv2
import numpy as np
from src.pipeline import Pipeline

# Load the request data from the JSON file
with open('test/test_data/request.json', 'r') as request_file:
	request_data = json.load(request_file)

# Initialize the Pipeline with input image and analyzer response file
pipeline = Pipeline(request_data["args"]["input_image_1"], "test/test_data/analyser_response.json")

# Run the pipeline and get the output data
output_data = pipeline.run()

# Save results
import json
with open("output.json", "w") as f:
    json.dump(output_data, f, indent=4)
```

You can find a running test in `test/test.py`.

# Testing

To be done with pytest...

# Structure

The project is modularized as follows:

```plaintext
tma_face_services_face_enhancer/
├── src/
│   ├── preprocessing.py
│   ├── inference.py
│   ├── postprocessing.py
│   └── pipeline.py
├── config/
│   ├── models.py
│   ├── warp_template.py
├── tests/
│   ├── test.py
│   ├── test_data/
│   │   ├── analyser_response.json
│   │   ├── marie.jpeg
│   │   └── request.json
└── requirements.txt
```