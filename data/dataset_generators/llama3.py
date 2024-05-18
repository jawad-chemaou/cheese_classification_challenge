from pathlib import Path
from tqdm import tqdm
from .base import DatasetGenerator
import transformers
import torch
import pickle

class Llama3DatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=5,
        
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

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
                for i in range(0, num_images_per_prompt, self.batch_size):
                    batch = prompt[i : i + self.batch_size]
                    images = self.generator.generate(batch)
                    self.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
                pbar.close()
                

    def create_prompts(self, labels_names):
        with open("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/example_dict.pkl", "rb") as file:
            prompts = pickle.load(file)
        return prompts

        

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
