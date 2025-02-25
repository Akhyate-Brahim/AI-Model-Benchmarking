from diffusers import StableDiffusionPipeline
import torch

model_id = input("Enter model name : ")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

file_name = input("Enter file name : ")

prompt = "A heroic medieval knight in polished silver armor, standing in a misty forest at dawn. His helmet is adorned with intricate engravings, and a royal red cape flows behind him. Soft sunlight pierces through the fog, casting an ethereal glow on the scene."

image = pipe(prompt).images[0]  

image.save(file_name)

