import ollama
from ollama import generate
import json
import pandas as pd
import glob
from PIL import Image
from io import BytesIO

# Function to get image files from the folder
def get_image_files(folder_path):
    return glob.glob(f"{folder_path}/*.jpg")

# Function to process an image and generate a description
def process_image(label, image_file):
    print(f"\nProcessing {image_file}\n")
    with Image.open(image_file) as img:
        with BytesIO() as buffer:
            img.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()

    full_response = ''
    # Generate a description of the image
    for response in generate(
            model='llava-llama3:latest',
            prompt=f'This image features {label} cheese. Describe this image, focusing on the cheese. It is less important to describe all the details in the scene, and more important to describe them as they relate to the cheese.',
            images=[image_bytes],
            options = {
                "max_tokens": 20,
                "temperature": 0.6,
                "top_p": 1,
                "top_k": 30,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": ["\n"]
            },
            stream=True):
        # Print the response to the console and add it to the full response
        full_response += response['response']
    #print(full_response)

    return image_file, full_response

# Function to get captions as a string from a DataFrame
def return_string_captions(df):
    captions = ""
    for index, row in df.iterrows():
        captions += row['description'] + "\n"
    return captions

def generate_json(df):  
    # Save the DataFrame to a JSON file
    print(df.head())

    with open(f'metadata.jsonl', 'w') as outfile:
        
        
        for index, row in df.iterrows():
            # Create a dictionary for each row
            entry = {
                "file_name": row['image_file'].split("/")[-1],
                "prompt": row['description']
            }
            # Write the dictionary as a JSON string to the file
            json.dump(entry, outfile)
            outfile.write('\n')

# Main function to process images for a given cheese label and return the DataFrame and captions
def process_cheese_images(label, generate_json=False):
    df = pd.DataFrame(columns=['image_file', 'description'])
    local_dir = f"./val/{label.upper()}"
    image_files = get_image_files(local_dir)
    image_files.sort()

    for image_file in image_files:
        if image_file not in df['image_file'].values:
            image_file, description = process_image(label, image_file)
            df.loc[len(df)] = [image_file, description]
    if generate_json:
        generate_json(df)
    return df, return_string_captions(df)

print(process_cheese_images("PECORINO")[1])

