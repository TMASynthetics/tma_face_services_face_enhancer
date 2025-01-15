import cv2
import numpy as np
import onnxruntime as ort
import json
from config.warp_templates import WARP_TEMPLATES
from config.models import MODELS


def estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size):
	normed_warp_template = WARP_TEMPLATES.get(warp_template) * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_warp_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	return affine_matrix

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, warp_template, crop_size):
	affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size)
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
	return crop_vision_frame, affine_matrix

def create_static_box_mask(crop_size, face_mask_blur, face_mask_padding):
	blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
	blur_area = max(blur_amount // 2, 1)
	box_mask = np.ones(crop_size).astype(np.float32)
	box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
	box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
	box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
	box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
	if blur_amount > 0:
		box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
	return box_mask

def forward_occlude_face(prepare_vision_frame):
	model_path = MODELS["face_occluder"]["path"]
	session = ort.InferenceSession(model_path)
	occlusion_mask = session.run(None,
		{
			'input': prepare_vision_frame
		})[0][0]

	return occlusion_mask

def create_occlusion_mask(crop_vision_frame):
	model_size = MODELS["face_occluder"]["size"]
	prepare_vision_frame = cv2.resize(crop_vision_frame, model_size)
	prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=(0)).astype(np.float32) / 255
	# print(prepare_vision_frame.shape)
	# print(prepare_vision_frame)
	# prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
	occlusion_mask = forward_occlude_face(prepare_vision_frame)
	occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
	occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
	occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
	return occlusion_mask

def prepare_crop_frame(crop_vision_frame):
	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
	crop_vision_frame = np.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
	return crop_vision_frame

def forward(crop_vision_frame):
	model_path = MODELS["gfpgan_1.4"]["path"]
	session = ort.InferenceSession(model_path)
	crop_vision_frame = session.run(None,
		{
			'input': crop_vision_frame
		})[0][0]

	return crop_vision_frame

def blend_frame(temp_vision_frame, paste_vision_frame):
	face_enhancer_blend = 1 - (0.8)
	temp_vision_frame = cv2.addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0)
	return temp_vision_frame

def normalize_crop_frame(crop_vision_frame):
	crop_vision_frame = np.clip(crop_vision_frame, -1, 1)
	crop_vision_frame = (crop_vision_frame + 1) / 2
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
	crop_vision_frame = (crop_vision_frame * 255.0).round()
	crop_vision_frame = crop_vision_frame.astype(np.uint8)[:, :, ::-1]
	return crop_vision_frame

def paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix):
	inverse_matrix = cv2.invertAffineTransform(affine_matrix)
	temp_size = temp_vision_frame.shape[:2][::-1]
	inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
	inverse_vision_frame = cv2.warpAffine(crop_vision_frame, inverse_matrix, temp_size, borderMode = cv2.BORDER_REPLICATE)
	paste_vision_frame = temp_vision_frame.copy()
	paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
	paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
	paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
	return paste_vision_frame, inverse_mask

if __name__ == '__main__':
	# Load the image
	target_vision_frame = cv2.imread('test/test_data/marie.jpeg')

	# load results from the face analyzer
	with open('analyser_response.json', 'r') as file:
		analyser_response = json.load(file)
	
	# As input, the face enhancer requires a face image and the face landmarks of the face in the image.
	temp_vision_frame = np.array(analyser_response['output_image_data']["bounding_box"])
	face_landmark_5 = np.array(analyser_response['output_image_data']["landmark_set"]["5"])
	print(temp_vision_frame.shape)

	model_template = MODELS["gfpgan_1.4"]['template']
	model_size = MODELS["gfpgan_1.4"]['size']

	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(target_vision_frame, face_landmark_5,  model_template, model_size)
	# cv2.imwrite('test/test_data/marie_enhanced.jpeg', crop_vision_frame)
	print(crop_vision_frame.shape)
	box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], 0.3, [0,0,0,0])
	crop_masks = [
		box_mask
	]
	# Do we want the choice between occlusion and region?
	occlusion_mask = create_occlusion_mask(crop_vision_frame)
	crop_masks.append(occlusion_mask)

	crop_vision_frame = prepare_crop_frame(crop_vision_frame)
	crop_vision_frame = forward(crop_vision_frame)
	crop_vision_frame = normalize_crop_frame(crop_vision_frame)
	crop_mask = np.minimum.reduce(crop_masks).clip(0, 1)
	paste_vision_frame, temp_vision_frame_mask = paste_back(target_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	temp_vision_frame = blend_frame(target_vision_frame, paste_vision_frame)
    
	output_data = {
		"temp_vision_frame": temp_vision_frame.tolist(),
		"temp_vision_frame_mask": temp_vision_frame_mask.tolist()
	}

	with open('output_data.json', 'w') as outfile:
		json.dump(output_data, outfile)
	
	# cv2.imshow('Temp Vision Frame', temp_vision_frame)
	# cv2.imshow('Temp Vision Frame Mask', temp_vision_frame_mask)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()