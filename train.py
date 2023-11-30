from utils import *
from sam_lora import *



sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
lora_sam = LoRA_Sam(sam,r = 4)
result = lora_sam.sam.image_encoder(torch.rand(size=(1,3,1024,1024)))
print(result.shape)