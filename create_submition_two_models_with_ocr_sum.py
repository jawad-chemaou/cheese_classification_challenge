import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
import pickle
import numpy as np

with open("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/ocr_predictions.pkl", "rb") as file:
    OCR_predictions = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        ocr_prediction_dict = OCR_predictions[image_name]
        self.class_names = sorted(os.listdir("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/dataset/train/llama3_prompts"))
        ocr_prediction = np.zeros(len(self.class_names))
        for cheese in ocr_prediction_dict.keys():
            if cheese in self.class_names:
                class_index = self.class_names.index(cheese)
                ocr_prediction[class_index] = ocr_prediction_dict[cheese]
        
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, ocr_prediction, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


@hydra.main(config_path="configs/train", config_name="config_two_datasets")
def create_submission(cfg):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model1 from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    
    model_bis = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path_bis)
    print(f"Loading model2 from checkpoint: {cfg.checkpoint_path_bis}")
    model_bis.load_state_dict(checkpoint)
    
    class_names = sorted(os.listdir(cfg.dataset.train_path))
    
    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        images, ocr_prediction_list, image_names = batch
        images = images.to(device)
        preds1 = model(images)
        preds2 = model_bis(images)
        preds = preds1 + preds2
            
        preds = torch.softmax(preds, dim=1)  # Get probabilities
        preds = preds.cpu().detach().numpy()
        
        for image_name, pred_probs, ocr_prediction in zip(image_names, preds, ocr_prediction_list):
            for i in range(len(ocr_prediction)):
                pred_probs[i] += ocr_prediction[i]
            final_label_index = pred_probs.argmax()
            final_label = class_names[final_label_index]
            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"id": [image_name], "label": [final_label]}),
                ]
            )
            
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)


if __name__ == "__main__":
    create_submission()
