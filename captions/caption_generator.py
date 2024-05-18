from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

def generate_caption(label):
    str = ""
    for i in range(0,25):
        if i < 10:
            image = Image.open(f"/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/dataset/val/{label}/00000{i}.jpg")
        else :
            image = Image.open(f"/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/dataset/val/{label}/0000{i}.jpg")

        #prompt = "Question: how many cats are there? Answer:"
        inputs = processor(images=image, return_tensors="pt").to(device="cuda", dtype=torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        str += generated_text + "\n"
    
    return str

print(generate_caption("CHEDDAR"))