MODELS_DIR_PATH = "/home/quillaur/HDD-1TO/models/bethel"

MODELS = {
    'gfpgan_1.4':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx',
        'path': f'{MODELS_DIR_PATH}/gfpgan_1.4.onnx',
        'template': 'ffhq_512',
        'size': (512, 512)
    },
    'face_occluder':
	{
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.onnx',
        'path': f'{MODELS_DIR_PATH}/dfl_xseg.onnx',
        'size': (256, 256)
	},
}