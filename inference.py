from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Load the test data
test_datagen = ImageDataGenerator(rescale=1./255)
model = load_model(r'c:\Users\sudha\OneDrive\Desktop\Revanth_project\resnet50_classification_model.h5') # Specify the trained model path
test_dir = r'c:\Users\sudha\OneDrive\Desktop\Revanth_project\data\test/'  # Replace with your test directory path

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',  # For multi-class classification
    shuffle=False  # Don't shuffle test data to align true labels with predictions
)

# Predict on the test set
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get true class labels
true_classes = test_generator.classes

# Calculate Precision, Recall, and F1 Score
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Calculate F1 Score per class (optional)
f1_per_class = f1_score(true_classes, predicted_classes, average=None)
print(f"F1 Score per class: {f1_per_class}")



# Get the true labels and predicted labels
true_classes = test_generator.classes
predicted_classes = np.argmax(model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size), axis=1)

# Compute confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.savefig('confusion_matrix.png')
