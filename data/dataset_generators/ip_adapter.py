from pathlib import Path
from tqdm import tqdm
from .base import DatasetGenerator
from captions.pecorino_dictionary_gpt4_prompts import get_prompts
import transformers
import torch
import pickle
from diffusers.utils import load_image

class IpAdapterGenerator(DatasetGenerator):
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
        for label in labels_names:
            image_id_0 = 0
            for i in range(0, 25):
                pbar = tqdm(range(0, 1, self.batch_size))
                pbar.set_description(
                f"Generating images for image: {label}/00000{i}.jpg"
                )
                
                if i < 10:
                    input_path = f"{self.input_dir}/{label}/00000{i}.jpg"
                else:
                    input_path = f"{self.input_dir}/{label}/0000{i}.jpg"
                image = load_image(input_path)

                for i in range(0, self.num_images_per_label, self.batch_size):
                    images = self.generator.generate(image)
                    self.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
            pbar.close()
            

        
        

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
