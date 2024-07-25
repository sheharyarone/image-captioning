import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import asyncio

async def image_caption(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
    model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

    
    raw_image = Image.open(image_path).convert('RGB')

    text = "a picture of "
    inputs = processor(raw_image, text, return_tensors="pt").to(device)

    out = model.generate(**inputs, num_beams = 3)
    await asyncio.sleep(1)
    return (processor.decode(out[0], skip_special_tokens=True))
