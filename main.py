import streamlit as st
import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

# Paths
BASE_DIR = "streamlit_dual_model_app"
TRAIN_DIR = os.path.join(BASE_DIR, "training_data")
TEST_DIR = os.path.join(BASE_DIR, "test_image")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_model.pt")
TF_MODEL_PATH = os.path.join(MODEL_DIR, "tf_model.h5")

for path in [TRAIN_DIR, TEST_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

# Streamlit Title
st.title("🧠 Image Classifier with Advanced Augmentation")

# Sidebar: Upload Training Images
st.sidebar.header("1. Upload Training Images (e.g., Apple_1.jpg, Banana_2.jpg)")
files = st.sidebar.file_uploader("Upload training images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if files:
    for file in files:
        label = file.name.split("_")[0]
        label_folder = os.path.join(TRAIN_DIR, label)
        os.makedirs(label_folder, exist_ok=True)
        with open(os.path.join(label_folder, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success("✅ Images uploaded and sorted!")

# Sidebar: Model and Epoch selection
framework = st.sidebar.selectbox("Choose Framework", ["PyTorch", "TensorFlow"])
epochs = st.sidebar.slider("Epochs", 3, 30, 10)

# Train PyTorch Model with Augmentation
# ✅ PyTorch Model Training (Improved)
def train_pytorch_model():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    class_names = dataset.classes

    model = efficientnet_b0(pretrained=True)

    # Fine-tune ALL layers
    for param in model.parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        st.write(f"[PyTorch] Epoch {epoch+1}/{epochs} — Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), PYTORCH_MODEL_PATH)
    return class_names

# Train TensorFlow Model with Augmentation
# ✅ TensorFlow Model Training (Improved)
def train_tf_model():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1
    )

    train_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    class_names = list(train_data.class_indices.keys())

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # ✅ Fine-tune all layers
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    model.fit(train_data, epochs=epochs, verbose=1)
    model.save(TF_MODEL_PATH)
    return class_names

# Train button
if st.sidebar.button("🔧 Train Model"):
    if not os.listdir(TRAIN_DIR):
        st.sidebar.warning("Please upload training images.")
    else:
        if framework == "PyTorch":
            class_labels = train_pytorch_model()
        else:
            class_labels = train_tf_model()
        st.sidebar.success("✅ Model trained!")

# Section 2: Upload test image
st.header("2. Upload Test Image")
test_img = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

if test_img:
    image = Image.open(test_img).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if framework == "PyTorch" and os.path.exists(PYTORCH_MODEL_PATH):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
        class_labels = dataset.classes

        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_labels))
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
        model.eval()

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, dim=0)
            label = class_labels[pred.item()]
            conf_value = conf.item()

    elif framework == "TensorFlow" and os.path.exists(TF_MODEL_PATH):
        tf_img = image.resize((224, 224))
        tf_arr = np.expand_dims(np.array(tf_img), axis=0)
        tf_arr = preprocess_input(tf_arr)

        model = tf.keras.models.load_model(TF_MODEL_PATH)
        dataset = ImageDataGenerator().flow_from_directory(TRAIN_DIR, target_size=(224, 224))
        class_labels = list(dataset.class_indices.keys())

        preds = model.predict(tf_arr)[0]
        conf_value = float(np.max(preds))
        label = class_labels[int(np.argmax(preds))]

    else:
        st.error("❗ Please train the selected model first.")
        label, conf_value = "Error", 0.0

    st.subheader("📊 Prediction Result")
    st.json({
        "predicted_class": label
    })

# Clear all
if st.sidebar.button("🗑️ Clear All Data"):
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    st.sidebar.success("✅ Data cleared!")
