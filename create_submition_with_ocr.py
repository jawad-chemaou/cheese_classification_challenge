import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
from ocr import CheeseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform, cheese_classifier):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        self.cheese_classifier = cheese_classifier
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        transformed_image = self.test_transform(image)
        ocr_words = self.cheese_classifier.process_image(image_path)
        return transformed_image, ocr_words, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg):
    # Initialize the cheese classifier
    cheese_classifier = CheeseClassifier("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/list_of_cheese.txt")

    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform), cheese_classifier
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))
    
    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        images, ocr_words_list, image_names = batch
        images = images.to(device)
        preds = model(images)
        preds = torch.softmax(preds, dim=1)  # Get probabilities
        preds = preds.cpu().detach().numpy()

        for image_name, pred_probs, ocr_words in zip(image_names, preds, ocr_words_list):
            matched_cheeses = cheese_classifier.classify_cheeses(ocr_words)
            if matched_cheeses:
                for cheese, confidence in matched_cheeses.items():
                    if cheese in class_names:
                        class_index = class_names.index(cheese)
                        pred_probs[class_index] += confidence  # Boost the confidence for OCR-detected cheese

            final_label_index = pred_probs.argmax()
            final_label = class_names[final_label_index]

            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"id": [image_name], "label": [final_label]}),
                ]
            )
    submission.to_csv(f"{cfg.root_dir}/submission_ocr.csv", index=False)


if __name__ == "__main__":
    create_submission()