import cv2
import numpy as np
from typing import Dict, Any, Tuple, List

class Postprocessing:
    def __init__(self, target_vision_frame: Any, crop_vision_frame: Any, crop_mask: Any, affine_matrix: Any):
        """
        Initialize the Postprocessing class with the given parameters.

        Args:
            target_vision_frame (Any): The target vision frame.
            crop_vision_frame (Any): The cropped vision frame.
            crop_mask (Any): The mask for the cropped vision frame.
            affine_matrix (Any): The affine transformation matrix.
        """
        self.target_vision_frame = target_vision_frame
        self.crop_vision_frame = crop_vision_frame
        self.crop_mask = crop_mask
        self.affine_matrix = affine_matrix
    
    def paste_back(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Paste the cropped vision frame back onto the temporary vision frame using the affine matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The pasted vision frame and the inverse mask.
        """
        # Invert the affine transformation matrix
        inverse_matrix = cv2.invertAffineTransform(self.affine_matrix)
        
        # Get the size of the temporary vision frame
        temp_size = self.target_vision_frame.shape[:2][::-1]
        
        # Warp the crop mask using the inverse affine matrix and clip values to [0, 1]
        inverse_mask = cv2.warpAffine(self.crop_mask, inverse_matrix, temp_size).clip(0, 1)
        
        # Warp the cropped vision frame using the inverse affine matrix
        inverse_vision_frame = cv2.warpAffine(self.crop_vision_frame, inverse_matrix, temp_size, borderMode=cv2.BORDER_REPLICATE)
        
        # Create a copy of the temporary vision frame to paste the cropped vision frame onto
        paste_vision_frame = self.target_vision_frame.copy()
        
        # Blend the inverse vision frame with the temporary vision frame using the inverse mask
        paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * self.target_vision_frame[:, :, 0]
        paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * self.target_vision_frame[:, :, 1]
        paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * self.target_vision_frame[:, :, 2]
        
        return paste_vision_frame, inverse_mask

    def blend_frame(self, paste_vision_frame: np.ndarray) -> np.ndarray:
        """
        Blend the pasted vision frame with the temporary vision frame.

        Args:
            paste_vision_frame (np.ndarray): The pasted vision frame.

        Returns:
            np.ndarray: The blended vision frame.
        """
        # Define the blend factor for the face enhancer
        face_enhancer_blend: float = 1 - 0.8
        
        # Blend the two frames using the specified blend factor
        temp_vision_frame = cv2.addWeighted(self.target_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0)
        
        return temp_vision_frame

    def run(self) -> Dict[str, List]:
        # Paste the cropped vision frame back onto the target vision frame
        paste_vision_frame, temp_vision_frame_mask = self.paste_back()
        
        # Blend the pasted vision frame with the target vision frame
        temp_vision_frame = self.blend_frame(paste_vision_frame)
        
        # Prepare the output data
        output_data: Dict[str, List] = {
            "temp_vision_frame": temp_vision_frame.tolist(),
            "temp_vision_frame_mask": temp_vision_frame_mask.tolist()
        }

        return output_data