import cv2
import json
import numpy as np
from typing import Tuple

class Preprocessing:
    def __init__(self, image_path: str, response_path: str) -> None:
        # Initialize the Preprocessing class with the paths to the image and response files
        self.image_path: str = image_path
        self.response_path: str = response_path

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        # Read the image from the specified path
        target_vision_frame: np.ndarray = cv2.imread(self.image_path)

        # Read the face analyzer response from a file
        with open(self.response_path, 'r') as file:
            analyser_response: dict = json.load(file)
        
        # Extract the face landmark 5 from the analyzer response
        face_landmark_5: np.ndarray = np.array(analyser_response['output_image_data']["landmark_set"]["5"])
        
        # Return the image and the face landmark 5
        return target_vision_frame, face_landmark_5