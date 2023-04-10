import numpy as np
import cv2
from segment_anything import build_sam_vit_h, SamPredictor
import onnxruntime

# Image 
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Loading model
sam = build_sam_vit_h()
predictor = SamPredictor(sam)

# Inputs 
input_point = np.array([[500, 875]])
input_label = np.array([1])
onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]

predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)


ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32),
    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
    "has_mask_input":  np.zeros(1, dtype=np.float32),
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

onnx_model_path = "sam_onnx_quantized_example.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

masks, _, low_res_logits = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

def show_mask(mask):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print(h, w, mask_image)
    
show_mask(masks)