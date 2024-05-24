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
                These are 25 captions for CHÈVRE cheese images:
                a table showasing many chèvre cheeses, over a table with a white tablecloth
                four pieces of identical chèvre cheese on a wooden board
                a logo for a chèvre cheese brand with the words chabris on it
                a piece of chèvre cheese with wiggly texture is cut in half and next to a glass and a fork over a table
                a whole chèvre buche on a white table with a piece of chèvre next to it
                a wooden cutting board with numerous types of chèvre cheese, a knife, crackers and a fig
                two small white cheese balls with wiggly structure sit on a wooden table
                a poster with different types of cheese and wine
                a piece of chèvre cheese with a sprig of rosemary
                a piece of chèvre with grapes and nuts on it
                a bowl of cream cheese with a spoon
                a round of cheese on a wooden cutting board
                a plate of food with sesame seeds and cream
                a bucket of paint with the words pepin and fils on it
                a display of cheese and other foods
                a cheese with a slice cut out of it
                a cheese board with some cheese and rosemary
                a piece of cheese is sitting on top of a piece of paper
                various types of cheese are displayed on display
                chevre cheese in a plastic container
                a piece of cheese on a cutting board with a knife
                a piece of cheese on a slate board with red peppers
                cheese and herbs on a piece of bread
                chèvre is shown on a table with other cheeses
                cheese is being made in a factory

                Can you generate 15 relevant prompts for CHÈVRE cheese?
                These should respect the composition of the 25 captions.
                Focus on the lightning, angle, and background.
                Make sure to showcase all different types of CHÈVRE cheese, and not only BUCHE DE CHEVRE.
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

labels = ["CHÈVRE"]
generator = Llama3PromptGenerator()
prompts = generator.create_prompts("labels")

with open("single_label.pkl", "wb") as file:
    pickle.dump(prompts, file)

