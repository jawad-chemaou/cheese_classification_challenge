from pathlib import Path
from tqdm import tqdm
import transformers
import torch
import pickle
#import llava_caption_generator

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
        #captions = llava_caption_generator.process_cheese_images(label)[1]
        #print(captions)
        messages = [
            {"role": "system", "content": "You are an image prompt generator which answers with detailed prompts to be directly used on a text-to-image model. You must only give the prompts as an answer, with a single entry space between prompts. The text should be formated as such: 'prompt1\nprompt2\nprompt3'"},
            {"role": "user", "content": f"""
                These are 25 captions for {label} cheese images:
                
                mage presents a scene featuring two pieces of Pecorino cheese on a wooden surface. The larger piece is positioned to the left and has been partially cut into, revealing its creamy interior. The smaller piece is located to the right and remains intact, with no visible cuts or damage. Both pieces exhibit a rich yellow color, indicative of their matured state. The cheese appears smooth and well-ripened, suggesting it's ready for consumption. The wooden surface underneath provides a natural and rustic backdrop to the cheese, enhancing its visual appeal. There are no discernible texts or other objects in the image. The relative positions of the cheese pieces and their interaction with the wooden surface are the main focus of this image.
The image captures a scene from a cheese cellar, where the main focus is on two types of Pecorino cheese. The first type, located on the left side of the frame, is stacked neatly in round wooden crates. Each crate contains multiple pieces of cheese, their golden-brown rinds hinting at their ripeness
The image presents a scene featuring the Pecorino cheese. The main focus is on two pieces of cheese, one large and one small, both exhibiting a pale yellow hue indicative of their freshness. The larger piece of cheese has a black label affixed to it, bearing text in white and red colors. This label displays the name "Pecorino Toscano DOP", which suggests the origin of this particular variety of Pecorino cheese. Additionally, the label shows an image of a cow, possibly symbolizing the agricultural source of the product
The image showcases a packet of Pecorino Romano D.O.P. cheese, which is the main subject of the image. The packet is white with a red logo on it, and it has some text written in black. The text reads "Vivaldi passion for Italian food pecorino romano d.o.p. Grattugiato".
The image showcases a large block of Pecorino cheese, resting on a white background. The cheese is pale yellow in color, with dark spots scattered throughout its surface. These spots are likely the result of the natural aging process of the cheese. On the right side of the cheese, there's a small bite taken out of it, revealing the creamy interior. The rest of the cheese remains untouched, maintaining its round shape and smooth texture. There is no text or other objects present in the image. The focus is solely on the Pecorino cheese, highlighting its unique characteristics and qualities.
The image presents a delightful scene of a culinary preparation. At the center of the frame, there's a piece of bread, cut into a triangle shape and topped with a slice of Pecorino cheese. The cheese exhibits a rich, orange hue, suggesting it might be freshly grated. Resting on the knife is another piece of this delicious cheese, indicating that more may have been used in its preparation
The image presents a scene of various food items arranged on a wooden table. The main focus is on two blocks of Pecorino cheese. One block is whole, while the other has been sliced into, revealing its rich and creamy texture
The image presents a scene of culinary preparation. At the center of the composition is a block of **PECORINO cheese**, exhibiting a pale yellow hue indicative of its fresh and aged nature. The cheese appears to be of high quality, suggesting it may have been used in professional settings
The image presents a scene of simplicity and elegance, featuring Pecorino cheese as the star. The cheese is a pale yellow color, its surface dotted with small holes that are characteristic of aged cheeses. It rests on a wooden cutting board, which adds a rustic charm to the scene
The image presents a slice of Pecorino cheese, exhibiting a pale yellow hue. The cheese has a crumbly texture and is speckled with small holes throughout its surface. It rests on a wooden cutting board that also holds a knife with a red handle. Scattered around the cheese are crumbs, adding to the rustic charm of the scene
The image presents a close-up view of two round Pecorino cheese balls, each adorned with a purple and pink sticker. The stickers feature an illustration of a woman's face and the word "Pecorino", indicating the brand of the cheese. Each cheese ball is tied together with white string, suggesting they might be sold in pairs
This image captures a simple yet enticing scene of three pieces of Pecorino cheese. The cheese, characterized by its golden yellow hue and visible holes, is the star of this composition
The image presents a single, round block of Pecorino cheese. The cheese has a golden brown color, indicating it is well aged and possibly even smoked. It is resting on a wooden cutting board with a circular shape and a handle on one side, suggesting that the cheese might be freshly cut or ready to be sliced. The background of the image is white, which contrasts with the rich color of the cheese and makes it stand out. There are no other objects in the image, and no text is visible. The relative position of the cheese to the cutting board suggests that it has been placed there for display or preparation.
The image presents a delightful culinary scene. At the center of the composition is a **black plate** holding six pieces of **pasta**. Each piece of pasta is generously coated with a layer of **green pesto**, which contrasts beautifully with the pasta's golden hue. Adding a pop of color and hinting at the flavors within, there are also **red cherry tomatoes** scattered atop the pesto
The image captures a collection of Pecorino cheese packages neatly arranged in a stack. Each package is wrapped in brown paper and secured with white strings, giving them an organized and appealing appearance. A blue label adorns each package, proudly displaying the name "Pecorino" along with an image of a train, perhaps indicating the origin or production method of the cheese. The background, though blurred, reveals the presence of other packages, suggesting that this stack is part of a larger collection or storage area.
The image showcases a collection of Pecorino cheese, presented in various forms. Dominating the scene is a large wheel of Pecorino cheese, its round shape standing out against the stark white background. The wheel itself is brown and yellow, indicative of its aged nature
The image presents a rustic scene set on a wooden table. At the center of the frame, a block of PECORINO cheese commands attention. It's resting on a wooden cutting board, its golden hue contrasting with the dark wood. The cheese appears to be freshly cut, as evidenced by the knife that lies next to it on the board
                                
                Can you generate 15 relevant prompts for {label} cheese?
                These should respect the composition of the 25 captions and be creative.
                For context: 
Pecorino cheese is a hard, crumbly cheese with a distinct, grainy texture and a pale, off-white color. It often features a rugged, natural rind and can range from a pale ivory to a deeper golden hue depending on its age. Shavings or chunks reveal a dense, slightly oily interior with tiny, irregular holes. Pecorino is typically grated over pasta, adding a sharp, salty flavor, or served in rustic slices on cheese boards, paired with fruits, nuts, and honey.
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


        

generator = Llama3PromptGenerator()
#prompts = generator.create_prompts(['CABECOU', 'PECORINO', 'VACHERIN', 'TOMME DE VACHE', 'CHEDDAR', 'MUNSTER', 'CHÈVRE', 'COMTÉ', 'BÛCHETTE DE CHÈVRE'])
prompts = generator.create_prompts(['PECORINO'])
with open("labels_llava_pecorino.pkl", "wb") as file:
    pickle.dump(prompts, file)
    