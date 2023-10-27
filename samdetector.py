import numpy as np
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
sam_checkpoint = res.find('other/sam_vit_h_4b8939.pth')
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

#________________ SAM ____________________________________
def do_sam(IMAGE_PATH, input_points, input_labels):
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    return masks, scores


def sam_show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def sam_create_mask_image(mask, output_path):
    color = np.array([255,0,0,1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    cv2.imwrite(output_path, mask_image)


#________________ FAST SAM ____________________________________
def do_fast_sam(IMAGE_PATH, input_points, input_labels):
    # input_point = np.array([[x, y]])
    # input_label = np.array([1])

    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, conf=0.20, iou=0.6, )
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    ann = prompt_process.point_prompt(points=input_points, pointlabel=input_labels)

    return prompt_process, ann


def fastsam_create_mask_image(ann, output_path):
    # New array to store RGB values
    shape = ann.shape
    mask = np.zeros((shape[1], shape[2], 3), dtype=np.uint8)

    # Generate a random RGB color
    random_color = np.array([255,0,0])

    # Replace False values with black and True values with random color
    mask[ann.squeeze()] = random_color
    mask[~ann.squeeze()] = [0, 0, 0]

    cv2.imwrite(output_path, mask)

    return mask