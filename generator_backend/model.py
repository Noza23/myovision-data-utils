from . import state
from segment_anything import sam_model_registry
import torch

def set_model(model_name: str, model_path: str):
    if state.MODEL is not None:
        raise Exception("Model already loaded")
    try:
        state.MODEL = sam_model_registry[model_name](model_path)
        if torch.cuda.is_available():
            state.MODEL.to(device="cuda:0")
            print("Model loaded to GPU")
 
    except Exception as e:
        raise Exception("Loading model failed: " + str(e))
    print("Model loaded successfully")
