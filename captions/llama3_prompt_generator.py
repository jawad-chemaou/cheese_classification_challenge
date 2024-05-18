from pathlib import Path
from tqdm import tqdm
import transformers
import torch
import pickle
from gpt3_5_base_prompts import get_prompts
from caption_generator import generate_caption

class Llama3PromptGenerator():
    def __init__(
    self,):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.access_token = "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"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
    def generate_prompts(self, label):
        """
        This method should generate prompts for a given label.
        """
        

        messages = [
            {"role": "system", "content": "You are an image prompt generator which answers with detailed prompts to be directly used on a text-to-image model. You must only give the prompts as an answer, with a single entry space between prompts."},
            {"role": "user", "content": f"""
                These are 25 captions for {label} cheese images:
                {generate_caption(label)}
                Do not emphasize on food.
                Can you generate 15 relevant prompts for {label} cheese?
                These should respect the composition of the 25 captions.
                Most of the prompts are simple and only represent the cheese, but change by their lightning, angle, and background.
                Make the other prompts original and adequate for each cheese, and focus on the context.
                All prompts should mention the cheese name.
                """
            }
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=600,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        prompt_list = outputs[0]["generated_text"][len(prompt):].split("\n")
        print(prompt_list)
        return prompt_list
    
    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            # ignore if string too small (meaning it's an incomplete prompt or a '\n')
            label_prompts = self.generate_prompts(label)
            prompts[label] = [{"prompt": prompt, "num_images": 10} for prompt in label_prompts if len(prompt) > 10]
            print(prompts[label])
        return prompts

with open("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/list_of_cheese.txt", "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
generator = Llama3PromptGenerator()
prompts = generator.create_prompts(labels)

with open("example_dict.pkl", "wb") as file:
    pickle.dump(prompts, file)

