from pathlib import Path
from tqdm import tqdm
import transformers
import torch

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
            {"role": "system", "content": "You are an image prompt generator which answers with detailed prompts to be directly used on a text-to-image model. You must only give the prompt as an answer."},
            {"role": "user", "content": f"""These are 10 relevant prompts for the BEAUFORT cheese:\n" \
                                A close-up shot of a slice of Beaufort cheese, showcasing its unique texture and color\n" \
                                Beaufort cheese on a rustic wooden background, accompanied by a glass of red wine, a baguette, and some grapes.\n" \
                                A whole wheel of Beaufort cheese in a traditional French cheese cellar, with a focus on the cheese's rind and natural aging process.\n" \
                                Beaufort cheese being grated over a steaming bowl of French onion soup, highlighting the cheese's melting quality.\n" \
                                A comparison shot of Beaufort cheese at different stages of aging, showcasing the transformation of its appearance and texture.\n" \
                                A plate of tartiflette, a classic French dish made with potatoes, bacon, and Beaufort cheese.\n" \
                                A croque monsieur sandwich made with Beaufort cheese and ham, served with a side of pommes frites.\n" \
                                A salad featuring mixed greens, cherry tomatoes, and Beaufort cheese, topped with a vinaigrette dressing.\n" \
                                A quiche Lorraine made with Beaufort cheese, eggs, and bacon, served with a fresh green salad.\n" \
                                A cheese board featuring Beaufort cheese, along with other French cheeses, charcuterie, and fresh fruit.\n" \
                                Can you generate 20 relevant prompt for {label} cheese, with various contexts, lightnings and point of views?"
                                """},
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
        print(outputs[0]["generated_text"][len(prompt):])
        return outputs[0]["generated_text"][len(prompt):]

generator = Llama3PromptGenerator()
print(generator.generate_prompts("PECORINO"))