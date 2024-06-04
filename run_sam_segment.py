#!/usr/bin/env python
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import torch
from tqdm import tqdm

# params
in_dir = 'test_images'
out_dir = 'sam_segmented_images'
#sam_model = "vit_l"
#sam_check = "checkpoint\\sam_vit_l_0b3195.pth"
sam_model = "vit_h"
sam_check = "checkpoint\\sam_vit_h_4b8939.pth"
#sam_model = "vit_b"
#sam_check = "checkpoint\\sam_vit_b_01ec64.pth"
device="cuda"
transparency = 0.3

# sam generator params
points_per_batch=64
points_per_side=64
pred_iou_thresh=0.86
stability_score_thresh=0.92
crop_n_layers=1
crop_n_points_downscale_factor=2
min_mask_region_area=100


def resize_image_by_width(image, wid=640):
    h, w, _ = image.shape
    new_w = wid
    new_h = int(h * (new_w / w))
    return cv2.resize(image, (new_w, new_h))


def process_image(img_path, out_path, mask_generator):
    image = cv2.imread(img_path)
    image = resize_image_by_width(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask generator wants the default uint8 image
    masks = mask_generator.generate(image_rgb)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(masks)
    annotated_image = mask_annotator.annotate(image, detections)
    cv2.imwrite(out_path, annotated_image)


if __name__ == "__main__":
    print("SAM model: " + sam_model)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sam = sam_model_registry[sam_model](checkpoint=sam_check)
    sam = sam.to(device=device)
    # sam = torch.compile(sam)
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=points_per_side,
                                               pred_iou_thresh=pred_iou_thresh,
                                               stability_score_thresh=stability_score_thresh,
                                               crop_n_layers=crop_n_layers,
                                               crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                               min_mask_region_area=min_mask_region_area)
    # process input directory
    for img in tqdm(os.listdir(in_dir)):
        print('processing ' + img)

        # change extension of output image to .png
        out_img = Path(img).stem + ".SAM_" + sam_model + ".sv.png"
        out_img = os.path.join(out_dir, out_img)

        # if we can read/decode this file as an image
        in_img = os.path.join(in_dir, img)
        if cv2.haveImageReader(in_img):
            process_image(in_img, out_img, mask_generator)
