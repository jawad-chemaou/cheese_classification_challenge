import pickle

# Specify the path to the pickle file
file_path = '/users/eleves-b/2022/jawad.chemaou/cheese_classification_challenge/example_dict.pkl'

# Open the pickle file in read mode
with open(file_path, 'rb') as file:
    # Load the contents of the pickle file
    example_dict = pickle.load(file)

# Now you can use the `example_dict` variable to access the data from the pickle file
print(example_dict['FETA'])