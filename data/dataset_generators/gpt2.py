from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from .base import DatasetGenerator


class GPT2DatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=5,
        lm_model_name="gpt2",
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.lm_model_name = lm_model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_model_name)
        self.lm_model = GPT2LMHeadModel.from_pretrained(lm_model_name, pad_token_id=self.tokenizer.eos_token_id)

    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            label_prompts = self.generate_prompts(label)
            prompts[label] = [{"prompt": prompt, "num_images": 1} for prompt in label_prompts]
        return prompts

    def generate_prompts(self, label):
        input_text = (
            "These are 10 relevant prompts for the BEAUFORT cheese:\n"
            "A close-up shot of a slice of Beaufort cheese, showcasing its unique texture and color\n"
            "Beaufort cheese on a rustic wooden background, accompanied by a glass of red wine, a baguette, and some grapes.\n"
            "A whole wheel of Beaufort cheese in a traditional French cheese cellar, with a focus on the cheese's rind and natural aging process.\n"
            "Beaufort cheese being grated over a steaming bowl of French onion soup, highlighting the cheese's melting quality.\n"
            "A comparison shot of Beaufort cheese at different stages of aging, showcasing the transformation of its appearance and texture.\n"
            "A plate of tartiflette, a classic French dish made with potatoes, bacon, and Beaufort cheese.\n"
            "A croque monsieur sandwich made with Beaufort cheese and ham, served with a side of pommes frites.\n"
            "A salad featuring mixed greens, cherry tomatoes, and Beaufort cheese, topped with a vinaigrette dressing.\n"
            "A quiche Lorraine made with Beaufort cheese, eggs, and bacon, served with a fresh green salad.\n"
            "A cheese board featuring Beaufort cheese, along with other French cheeses, charcuterie, and fresh fruit.\n"
            f"\nCan you generate {self.num_images_per_label} relevant prompts for {label} cheese, giving only the prompt and considering that in 70% of the times, you will generate a simple prompt similar to the first one you were presented?"
        )

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.lm_model.generate(input_ids, max_length=200, num_return_sequences=self.num_images_per_label)
        generated_prompts = [self.tokenizer.decode(output[i], skip_special_tokens=True) for i in range(self.num_images_per_label)]

        return generated_prompts


