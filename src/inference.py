import numpy as np
import cv2
import onnxruntime as ort
from config.models import MODELS
from config.warp_templates import WARP_TEMPLATES
from typing import Tuple, List

class Inference:
    def __init__(self, target_vision_frame: np.ndarray, face_landmark_5: np.ndarray):
        """
        Initialize the Inference class with the target vision frame and face landmarks.

        Args:
            target_vision_frame (np.ndarray): The input vision frame containing the face to be enhanced.
            face_landmark_5 (np.ndarray): The 5-point face landmarks used for face alignment and warping.
        """
        # Store the input vision frame containing the face to be enhanced
        self.target_vision_frame = target_vision_frame
        
        # Store the 5-point face landmarks used for face alignment and warping
        self.face_landmark_5 = face_landmark_5
    
    def estimate_matrix_by_face_landmark_5(self, warp_template: str, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Estimate the affine transformation matrix using the provided face landmarks and warp template.

        Args:
            warp_template (str): The template name used to retrieve the warp template from the configuration.
            crop_size (Tuple[int, int]): The desired size of the cropped vision frame (width, height).

        Returns:
            np.ndarray: The estimated affine transformation matrix.
        """
        # Retrieve the warp template from the configuration and normalize it by the crop size
        normed_warp_template = WARP_TEMPLATES.get(warp_template) * crop_size
        
        # Estimate the affine transformation matrix using the face landmarks and the normalized warp template
        # The method used is RANSAC with a reprojection threshold of 100
        affine_matrix = cv2.estimateAffinePartial2D(self.face_landmark_5, normed_warp_template, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
        
        # Return the estimated affine transformation matrix
        return affine_matrix
    
    def warp_face_by_face_landmark_5(self, warp_template: str, crop_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp the face in the given vision frame using the estimated affine matrix based on face landmarks.

        Args:
            warp_template (str): The template name used to retrieve the warp template from the configuration.
            crop_size (Tuple[int, int]): The desired size of the cropped vision frame (width, height).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the cropped vision frame and the affine transformation matrix.
        """
        # Estimate the affine transformation matrix using the provided face landmarks and warp template
        affine_matrix = self.estimate_matrix_by_face_landmark_5(warp_template, crop_size)
        
        # Warp the face in the input vision frame using the estimated affine matrix
        crop_vision_frame = cv2.warpAffine(self.target_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        
        # Return the cropped vision frame and the affine transformation matrix
        return crop_vision_frame, affine_matrix

    def create_static_box_mask(self, crop_size: Tuple[int, int], face_mask_blur: float, face_mask_padding: List[int]) -> np.ndarray:
        """
        Create a static box mask with optional blur and padding.

        Args:
            crop_size (Tuple[int, int]): The size of the cropped vision frame (width, height).
            face_mask_blur (float): The amount of blur to apply to the mask edges. 
                                    This is a fraction of the crop size.
            face_mask_padding (List[int]): Padding values for the mask in the order [top, right, bottom, left].
                                           These values are percentages of the crop size.

        Returns:
            np.ndarray: The generated static box mask with optional blur and padding applied.
                        The mask values are in the range [0, 1].
        """
        # Calculate the amount of blur to apply based on the crop size and blur factor
        blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        
        # Initialize the box mask with ones (fully visible)
        box_mask = np.ones(crop_size).astype(np.float32)
        
        # Apply padding to the top of the mask
        box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
        
        # Apply padding to the bottom of the mask
        box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
        
        # Apply padding to the left of the mask
        box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
        
        # Apply padding to the right of the mask
        box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
        
        # Apply Gaussian blur to the mask if blur amount is greater than 0
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        
        return box_mask

    def forward_occlude_face(self, prepare_vision_frame: np.ndarray) -> np.ndarray:
        """
        Runs the face occlusion model to generate an occlusion mask for the given input frame.

        Args:
            prepare_vision_frame (np.ndarray): A preprocessed frame containing the face to be occluded. 
                                               This should be a numpy array with the appropriate dimensions 
                                               and format expected by the face occlusion model.

        Returns:
            np.ndarray: A numpy array representing the occlusion mask for the input frame. The mask will 
                        have the same spatial dimensions as the input frame, with values indicating the 
                        occluded regions of the face.
        """
        # Run the face occlusion model to get the occlusion mask
        # Get the path to the face occlusion model from the configuration
        model_path = MODELS["face_occluder"]["path"]
        
        # Create an ONNX runtime inference session with the face occlusion model
        session = ort.InferenceSession(model_path)
        
        # Run the face occlusion model on the prepared vision frame and get the occlusion mask
        occlusion_mask = session.run(None, {'input': prepare_vision_frame})[0][0]
        
        # Return the occlusion mask generated by the model
        return occlusion_mask
    
    def create_occlusion_mask(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        """
        Create an occlusion mask for the given cropped vision frame.

        This function resizes the input frame to match the model's expected input size,
        normalizes the pixel values, and then uses a pre-trained model to generate an
        occlusion mask. The mask is then resized back to the original frame size and
        smoothed using a Gaussian blur.

        Args:
            crop_vision_frame (np.ndarray): The cropped vision frame for which to create the occlusion mask.
            Expected shape is (height, width, channels).

        Returns:
            np.ndarray: The generated occlusion mask with the same height and width as the input frame.
            The mask values are in the range [-1, 1].
        """
        # Create an occlusion mask for the cropped vision frame
        # Get the model input size from the configuration
        model_size = MODELS["face_occluder"]["size"]
        
        # Resize the cropped vision frame to match the model's expected input size
        prepare_vision_frame = cv2.resize(crop_vision_frame, model_size)
        
        # Expand dimensions to add a batch dimension and normalize pixel values to the range [0, 1]
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=0).astype(np.float32) / 255.0
        
        # Run the occlusion model to get the occlusion mask
        occlusion_mask = self.forward_occlude_face(prepare_vision_frame)
        
        # Transpose the occlusion mask dimensions from (1, H, W) to (H, W, 1) and clip values to [0, 1]
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        
        # Resize the occlusion mask back to the original cropped vision frame size
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        
        # Apply Gaussian blur to smooth the occlusion mask, clip values to [0.5, 1], and scale to [-1, 1]
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        
        return occlusion_mask
    
    def prepare_crop_frame(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        """
        Prepare the cropped vision frame for model input.
        
        Args:
            crop_vision_frame (np.ndarray): The cropped vision frame to be prepared.
        
        Returns:
            np.ndarray: The prepared cropped vision frame ready for model input.
        """
        # Convert the color channel order from BGR to RGB
        crop_vision_frame = crop_vision_frame[:, :, ::-1]
        
        # Normalize the pixel values to the range [0, 1]
        crop_vision_frame = crop_vision_frame / 255.0
        
        # Scale the pixel values to the range [-1, 1]
        crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
        
        # Transpose the dimensions from (H, W, C) to (C, H, W) and add a batch dimension
        crop_vision_frame = np.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        
        return crop_vision_frame

    def forward(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        """
        Run the face enhancement model on the prepared cropped vision frame.
        
        Args:
            crop_vision_frame (np.ndarray): The prepared cropped vision frame.
        
        Returns:
            np.ndarray: The output from the face enhancement model.
        """
        # Get the path to the face enhancement model from the configuration
        model_path = MODELS["gfpgan_1.4"]["path"]
        
        # Create an ONNX runtime inference session with the model
        session = ort.InferenceSession(model_path)
        
        # Run the model on the input cropped vision frame and get the output
        crop_vision_frame = session.run(None, {'input': crop_vision_frame})[0][0]
        
        return crop_vision_frame
    
    def normalize_crop_frame(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        """
        Normalize the cropped vision frame after model inference.
        
        Args:
            crop_vision_frame (np.ndarray): The output from the face enhancement model.
        
        Returns:
            np.ndarray: The normalized and converted cropped vision frame.
        """
        # Clip the values to be within the range [-1, 1]
        crop_vision_frame = np.clip(crop_vision_frame, -1, 1)
        
        # Scale the values from [-1, 1] to [0, 1]
        crop_vision_frame = (crop_vision_frame + 1) / 2
        
        # Transpose the dimensions from (C, H, W) to (H, W, C)
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        
        # Scale the values from [0, 1] to [0, 255] and round them
        crop_vision_frame = (crop_vision_frame * 255.0).round()
        
        # Convert the values to uint8 type and change the color channel order from RGB to BGR
        crop_vision_frame = crop_vision_frame.astype(np.uint8)[:, :, ::-1]
        
        return crop_vision_frame

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Main method to run the face enhancement pipeline

        # Get the model template and size from the configuration
        model_template = MODELS["gfpgan_1.4"]['template']
        model_size = MODELS["gfpgan_1.4"]['size']

        # Warp the face in the target vision frame using the face landmarks and model template
        crop_vision_frame, affine_matrix = self.warp_face_by_face_landmark_5(model_template, model_size)
        
        # Create a static box mask for the cropped vision frame
        box_mask = self.create_static_box_mask(crop_vision_frame.shape[:2][::-1], 0.3, [0, 0, 0, 0])
        crop_masks = [box_mask]

        # Create an occlusion mask for the cropped vision frame
        occlusion_mask = self.create_occlusion_mask(crop_vision_frame)
        crop_masks.append(occlusion_mask)

        # Prepare the cropped vision frame for model input
        crop_vision_frame = self.prepare_crop_frame(crop_vision_frame)

        # Run the face enhancement model on the prepared cropped vision frame
        crop_vision_frame = self.forward(crop_vision_frame)

        # Normalize the output from the model to get the final enhanced face image
        crop_vision_frame = self.normalize_crop_frame(crop_vision_frame)

        # Combine the static box mask and occlusion mask to get the final crop mask
        crop_mask = np.minimum.reduce(crop_masks).clip(0, 1)
        
        # Return the enhanced face image, the final crop mask, and the affine transformation matrix
        return crop_vision_frame, crop_mask, affine_matrix