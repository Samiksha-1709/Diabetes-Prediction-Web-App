import numpy as np
import pickle

try:
    # Loading the saved model
    model_path = 'D:/Deploying machine learning/trained_model.sav'
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Input data
    input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Making a prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Output result
    if prediction[0] == 0:
        print('The person is not diabetic')
    else:
        print('The person is diabetic')

except FileNotFoundError:
    print(f"Error: The model file was not found at {model_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")