from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load the test dataset
data_gen = ImageDataGenerator(rescale=1./255)
test_data = data_gen.flow_from_directory('minet', target_size=(64, 64), batch_size=32, class_mode='categorical')

# Load the models
standard_model = load_model('cnn_model.h5')
tuned_model = load_model('cnn_model_best.h5')

# Evaluate the models
standard_accuracy = standard_model.evaluate(test_data)[1]  # Accuracy is the second value in the output
tuned_accuracy = tuned_model.evaluate(test_data)[1]

# Print the results
print(f"standard CNN accuracy: {standard_accuracy * 100:.2f}%")
print(f"tuned CNN accuracy: {tuned_accuracy * 100:.2f}%")