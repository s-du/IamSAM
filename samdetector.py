import numpy as np
import matplotlib.pyplot as plt
import sys
from fastsam import FastSAM, FastSAMPrompt
from segment_anything import sam_model_registry, SamPredictor

import resources as res
import cv2

sys.path.append("..")

# Defining FASTsam model
sam_checkpoint = res.find('other/FastSAM-x.pt')
DEVICE = "cpu"
model = FastSAM(sam_checkpoint)

# Defining sam model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def do_sam(image, output_folder, x,y):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

def do_fast_sam(IMAGE_PATH, input_points, input_labels):
    # input_point = np.array([[x, y]])
    # input_label = np.array([1])

    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, conf=0.25, iou=0.7, )
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    ann = prompt_process.point_prompt(points=input_points, pointlabel=input_labels)

    return prompt_process, ann


def create_mask_image(ann, output):
    # New array to store RGB values
    shape = ann.shape

    print(shape)
    mask = np.zeros((shape[1], shape[2], 3), dtype=np.uint8)

    # Generate a random RGB color
    random_color = np.random.randint(0, 256, 3)

    # Replace False values with black and True values with random color
    mask[ann.squeeze()] = random_color
    mask[~ann.squeeze()] = [0, 0, 0]

    cv2.imshow('test', mask)
    cv2.waitKey(0)

    return mask