import re
import unicodedata
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fuzzywuzzy import fuzz, process
import os, pickle


import re, json
import unicodedata
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fuzzywuzzy import fuzz, process

class CheeseClassifier:
    def __init__(self, cheese_file):
        self.model = ocr_predictor(pretrained=True)
        self.aliases = self.load_aliases()
        self.cheese_names = self.load_cheese_names(cheese_file)
        
    def normalize_text(self, text):
        nfkd_form = unicodedata.normalize('NFKD', text)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper()

    def load_aliases(self):
        # Add your actual alias loading logic here
        aliases = {
            "BRIE DE MELUN": ['BRIE', 'MELUN'],
            "COMTÉ": "COMTE",
            "FETA": ["GREC", "GREQUE", "GREEK", "GREECE"],
            "FOURME D’AMBERT": ["FOURME", "AMBERT"],
            "FROMAGE FRAIS": ["FRAIS"],
            "GRUYÈRE": "GRUYERE",
            "MONT D’OR": ["MONT", "OR"],
            "OSSAU- IRATY": ["OSSAU", "IRATY", "OSSAU-IRATY"],
            "POULIGNY SAINT- PIERRE": ["POULIGNY", "PIERRE", "SAINT-PIERRE"],
            "SAINT- FÉLICIEN": ["FELICIEN", "SAINT-FELICIEN"],
            "SAINT-NECTAIRE": ["NECTAIRE", "SAINT-NECTAIRE"],
            "TÊTE DE MOINES": ["TETE", "MOINES", "MOINE"],
            "TOMME DE VACHE": ["TOMME", "VACHE"],
            ("CHÈVRE", "FETA", "CABECOU", "CHABICHOU", "BÛCHETTE DE CHÈVRE"): 'CHEVRE',
            ("CHÈVRE", "CABECOU"): 'CABECOU',
            ("CHÈVRE", "CHABICHOU"): 'CHABICHOU',
            ("CHÈVRE", "BÛCHETTE DE CHÈVRE"): ["BUCHE", "BUCHETTE"]
        }
        return aliases

    def load_cheese_names(self, cheese_file):
        with open(cheese_file, "r") as f:
            labels = f.readlines()
            labels = [label.strip().upper() for label in labels]
        return labels

    def map_aliases(self, word):
        for key, values in self.aliases.items():
            if isinstance(values, list):
                if word in values:
                    return key, True
            else:
                if word == values:
                    return key, True
        return word, False
    
    def process_image(self, image_path):
        image = DocumentFile.from_images(image_path)
        result = self.model(image)
        lines = self.normalize_text(result.render())
        lines = lines.strip().split('\n')
        filtered_lines = [line.strip() for line in lines if line.strip()]
        words = [word for line in filtered_lines for word in line.split()]

        corrected_words = [re.sub(r'0', 'O', word) for word in words]
        corrected_words = [re.sub(r'1', 'L', word) for word in corrected_words]
        
        return corrected_words
    
    def classify_cheeses(self, words):
        matched_cheese = {}

        for word in words:
            mapped_word, is_alias = self.map_aliases(word)
            if not is_alias:
                match, score = process.extractOne(mapped_word, self.cheese_names, scorer=fuzz.ratio)
                if score > 80:  # Threshold for fuzzy matching
                    matched_cheese[match] = score / 100  # Normalize the score to a range of 0 to 1
            else:
                if isinstance(mapped_word, tuple):
                    for mw in mapped_word:
                        matched_cheese[mw] = 1.0  # Highest confidence for exact matches
                else:
                    matched_cheese[mapped_word] = 1.0

        return matched_cheese
    
    def return_cheese_single_image(self, image_path):
        words = self.process_image(image_path)
        matched_cheeses = self.classify_cheeses(words)
        return matched_cheeses
    
    def return_cheeses_dir_path(self, dir_path):
        matched_cheeses = {}
        for image in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image)
            matched_cheeses[image] = self.return_cheese_single_image(image_path)
        return matched_cheeses
    
    def get_json_file(self, dir_path, json_file):
        matched_cheeses = self.return_cheeses_dir_path(dir_path)
        with open("ocr_predictions.pkl", "wb") as file:
            pickle.dump(matched_cheeses, file)
        return matched_cheeses

# Example usage
if __name__ == "__main__":
    classifier = CheeseClassifier("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/list_of_cheese.txt")
    matched_cheeses = classifier.get_json_file("/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/dataset/test", "/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/ocr_predictions.json")
    
    print("Matched Cheeses: ", matched_cheeses)