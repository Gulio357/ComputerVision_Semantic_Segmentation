from utils import *
from segment_anything import sam_model_registry

import pdb


sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

# NOTE: downsample
sam.image_encoder.downsample_pos_embed()

device = "cpu"

sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread('imgs/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

input_point = np.array([[500, 375]])
input_label = np.array([1])

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  


# pdb.set_trace()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  