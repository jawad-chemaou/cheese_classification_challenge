from .base import DatasetGenerator


class SimplePromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=5,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            prompts[label] = []
            prompts[label].append(
                {
                    "prompt": f"An image of a {label} cheese",
                    "num_images": self.num_images_per_label*4,
                }
            )
            prompts[label].append(
                {
                    "prompt": f"""
                    These are 10 relevant prompts for the BEAUFORT cheese:\n" \
                     "    A close-up shot of a slice of Beaufort cheese, showcasing its unique texture and color\n" \
                     "    Beaufort cheese on a rustic wooden background, accompanied by a glass of red wine, a baguette, and some grapes.\n" \
                     "    A whole wheel of Beaufort cheese in a traditional French cheese cellar, with a focus on the cheese's rind and natural aging process.\n" \
                     "    Beaufort cheese being grated over a steaming bowl of French onion soup, highlighting the cheese's melting quality.\n" \
                     "    A comparison shot of Beaufort cheese at different stages of aging, showcasing the transformation of its appearance and texture.\n" \
                     "    A plate of tartiflette, a classic French dish made with potatoes, bacon, and Beaufort cheese.\n" \
                     "    A croque monsieur sandwich made with Beaufort cheese and ham, served with a side of pommes frites.\n" \
                     "    A salad featuring mixed greens, cherry tomatoes, and Beaufort cheese, topped with a vinaigrette dressing.\n" \
                     "    A quiche Lorraine made with Beaufort cheese, eggs, and bacon, served with a fresh green salad.\n" \
                     "    A cheese board featuring Beaufort cheese, along with other French cheeses, charcuterie, and fresh fruit.\n" \
                     f"\n\nCan you generate a relevant prompt for {label} cheese, giving only the prompt?"
                    """,
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts
