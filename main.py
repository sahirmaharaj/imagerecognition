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
import re

# Paths
BASE_DIR = "streamlit_dual_model_app"
TRAIN_DIR = os.path.join(BASE_DIR, "training_data")
TEST_DIR = os.path.join(BASE_DIR, "test_image")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_model.pt")
TF_MODEL_PATH = os.path.join(MODEL_DIR, "tf_model.h5")

for path in [TRAIN_DIR, TEST_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

st.title("🧠 Image Classifier with Import/Export and Augmentation")

# Sidebar: Upload Training Images
st.sidebar.header("1. Upload Training Images")
files = st.sidebar.file_uploader("Upload training images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if files:
    valid_files = 0
    for file in files:
        filename = file.name.strip()
        name_part = os.path.splitext(filename)[0]
        label_match = re.match(r"([a-zA-Z0-9]+)", name_part)
        if label_match:
            label = label_match.group(1)
            label_folder = os.path.join(TRAIN_DIR, label)
            os.makedirs(label_folder, exist_ok=True)
            safe_filename = re.sub(r"[^\w\-_.]", "_", filename)
            with open(os.path.join(label_folder, safe_filename), "wb") as f:
                f.write(file.read())
            valid_files += 1
        else:
            st.warning(f"⚠️ Skipped invalid file: {filename}")
    st.sidebar.success(f"✅ Uploaded and sorted {valid_files} image(s).")

# Sidebar: Select framework and epochs
framework = st.sidebar.selectbox("Choose Framework", ["PyTorch", "TensorFlow"])
epochs = st.sidebar.slider("Epochs", 3, 30, 10)

# Sidebar: Upload Pretrained Model
st.sidebar.markdown("---")
st.sidebar.header("2. Import Pretrained Model")
uploaded_model = st.sidebar.file_uploader("Upload a model (.pt or .h5)", type=["pt", "h5"])
if uploaded_model:
    model_path = os.path.join(MODEL_DIR, uploaded_model.name)
    with open(model_path, "wb") as f:
        f.write(uploaded_model.read())
    if uploaded_model.name.endswith(".pt"):
        PYTORCH_MODEL_PATH = model_path
        st.sidebar.success("Using uploaded PyTorch model.")
    elif uploaded_model.name.endswith(".h5"):
        TF_MODEL_PATH = model_path
        st.sidebar.success("Using uploaded TensorFlow model.")

# Train PyTorch
def train_pytorch_model():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    class_names = dataset.classes

    model = efficientnet_b0(pretrained=True)
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

# Train TensorFlow
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
        TRAIN_DIR, target_size=(224, 224), batch_size=16, class_mode='categorical', subset='training'
    )
    class_names = list(train_data.class_indices.keys())

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    model.fit(train_data, epochs=epochs, verbose=1)
    model.save(TF_MODEL_PATH)
    return class_names

# Train model
if st.sidebar.button("🔧 Train Model"):
    if not os.listdir(TRAIN_DIR):
        st.sidebar.warning("Please upload training images.")
    else:
        if framework == "PyTorch":
            class_labels = train_pytorch_model()
        else:
            class_labels = train_tf_model()
        st.sidebar.success("✅ Model trained!")

# Always-show model download buttons
st.sidebar.markdown("---")
st.sidebar.header("3. Export Trained Model")

if framework == "PyTorch" and os.path.exists(PYTORCH_MODEL_PATH):
    with open(PYTORCH_MODEL_PATH, "rb") as f:
        st.sidebar.download_button("⬇️ Download PyTorch Model", f, file_name="model.pt")
elif framework == "TensorFlow" and os.path.exists(TF_MODEL_PATH):
    with open(TF_MODEL_PATH, "rb") as f:
        st.sidebar.download_button("⬇️ Download TensorFlow Model", f, file_name="model.h5")
else:
    st.sidebar.info("Train or upload a model to download it.")

# Upload test image
st.header("Upload Test Image")
test_img = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])
if test_img:
    image = Image.open(test_img).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if framework == "PyTorch" and os.path.exists(PYTORCH_MODEL_PATH):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    elif framework == "TensorFlow" and os.path.exists(TF_MODEL_PATH):
        tf_img = image.resize((224, 224))
        tf_arr = np.expand_dims(np.array(tf_img), axis=0)
        tf_arr = preprocess_input(tf_arr)

        model = tf.keras.models.load_model(TF_MODEL_PATH)
        dataset = ImageDataGenerator().flow_from_directory(TRAIN_DIR, target_size=(224, 224))
        class_labels = list(dataset.class_indices.keys())

        preds = model.predict(tf_arr)[0]
        label = class_labels[int(np.argmax(preds))]

    else:
        st.error("❗ Please train or import a model first.")
        label = "Error"

    st.subheader("📊 Prediction Result")
    st.json({"predicted_class": label})

# Clear all
if st.sidebar.button("🗑️ Clear All Data"):
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    st.sidebar.success("✅ Data cleared!")
