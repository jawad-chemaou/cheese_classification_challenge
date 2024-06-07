from pathlib import Path
from tqdm import tqdm
from .base import DatasetGenerator
from captions.pecorino_dictionary_gpt4_prompts import get_prompts
import transformers
import torch
import pickle
from diffusers.utils import load_image

class Img2ImgDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        input_dir="dataset/val",
        num_images_per_label=7,
        
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.input_dir = input_dir

    def generate(self, labels_names):
        labels_prompts = self.create_prompts(labels_names)
        for label, label_prompts in labels_prompts.items():
            image_id_0 = 0
            for prompt_metadata in label_prompts:
                num_images_per_prompt = prompt_metadata["num_images"]
                prompt = [prompt_metadata["prompt"]] * num_images_per_prompt
                pbar = tqdm(range(0, num_images_per_prompt, self.batch_size))
                pbar.set_description(
                    f"Generating images for prompt: {prompt_metadata['prompt']}"
                )
                for i in range(0, 25):
                    prompt = f"A close-up photograph of finely grated Pecorino cheese on a wooden cutting board, with a partially sliced block of Pecorino beside it, showcasing the contrast between the solid block and the delicate shreds."
                    input_path = f"{self.input_dir}/{label}/00000{i}.jpg"
                    image = load_image(input_path)

                    for i in range(0, num_images_per_prompt, self.batch_size):
                        images = self.generator.generate(prompt, image)
                        self.save_images(images, label, image_id_0)
                        image_id_0 += len(images)
                        pbar.update(1)
                pbar.close()
                

    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            prompts[label] = []
            prompts[label].append(
                {
                    "prompt": f"An image of {label} cheese.",
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts
        
        

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
