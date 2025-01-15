from src.preprocessing import Preprocessing
from src.inference import Inference
from src.postprocessing import Postprocessing
from typing import Dict

class Pipeline:
    def __init__(self, image_path: str, response_path: str) -> None:
        """
        Initialize the Pipeline with the given image and response paths.

        :param image_path: Path to the input image.
        :param response_path: Path to the response file.
        """
        self.image_path = image_path
        self.response_path = response_path

    def run(self) -> Dict:
        # Preprocessing
        preprocessing = Preprocessing(self.image_path, self.response_path)
        target_vision_frame, face_landmark_5 = preprocessing.run()

        # Inference
        inference = Inference(target_vision_frame, face_landmark_5)
        crop_vision_frame, crop_mask, affine_matrix = inference.run()

        # Postprocessing
        postprocessing = Postprocessing(target_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        outputs = postprocessing.run()

        return outputs
