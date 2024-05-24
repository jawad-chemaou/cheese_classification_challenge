import ollama
response = ollama.generate(model='llava-llama3', 
                           prompt = "/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/dataset/val/BEAUFORT/000002.jpg This image is labeled as BEAUFORT cheese. Do a text prompt for SDXL image, list all elements in this image in the text prompt as well. You must include the name of the cheese."
  )
print(response['response'])