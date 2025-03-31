import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# Define image size
IMG_SIZE = (224, 224)  # ResNet50 expects 224x224 input size

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Rescale pixel values to [0, 1]
    rotation_range=30,       # Random rotations
    width_shift_range=0.2,   # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,         # Random shear transformations
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Random horizontal flip
    fill_mode='nearest'      # Fill mode for new pixels
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation
# Directory paths
train_dir = r'C:\Users\sudha\OneDrive\Desktop\Revanth_project\data\train/'
val_dir = r'C:\Users\sudha\OneDrive\Desktop\Revanth_project\data\val/'

# Load images from the directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'  # since it's a multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)
# Load pre-trained ResNet50 model + top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Build your model on top of the base ResNet50 model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Global average pooling layer #for optimal loss @ convex graph)
    layers.Dense(512, activation='relu'),  # Fully connected layer #avoid vanishing gradient issue
    layers.Dense(6, activation='softmax')  # 5 classes, softmax for classification #for multi class else sigmoid
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']) #adaptive lr with momentum
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,  # You can adjust the number of epochs as needed
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Plot training & validation accuracy
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('train_val_loss_accuracy_graph.png')
#saving the trained model
model.save('resnet50_classification_model.h5') 
