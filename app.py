import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Utility Functions ---
def load_img(image_file):
    image = Image.open(image_file).resize((224, 224))
    image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.convert_to_tensor(image)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor, tf.uint8)
    return Image.fromarray(tensor.numpy()[0])

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# --- Define StyleContentModel class (same as used during training) ---
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        vgg = tf.keras.applications.VGG19(include_top=False, weights=None)
        vgg.load_weights("vgg19_finetuned_nst_model.h5", by_name=True, skip_mismatch=True)  # Load your fine-tuned weights
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
        self.vgg = tf.keras.Model([vgg.input], outputs)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)
        style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(output) for output in style_outputs]
        content_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}
        return {'style': style_dict, 'content': content_dict}

# --- Style Transfer Loss ---
def style_content_loss(outputs, style_targets, content_targets, style_weight=1e-2, content_weight=1e4):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([
        tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
        for name in style_outputs
    ])
    style_loss *= style_weight / len(style_outputs)
    content_loss = tf.add_n([
        tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
        for name in content_outputs
    ])
    content_loss *= content_weight / len(content_outputs)
    return style_loss + content_loss

@tf.function
def train_step(image, extractor, style_targets, content_targets, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets)
        loss += 30 * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

# --- Streamlit App ---
st.title("🎨 Neural Style Transfer (Fine-tuned VGG19)")
st.write("Upload a content image and a style image. Your trained model will perform the transfer.")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_image = load_img(content_file)
    style_image = load_img(style_file)

    st.image(Image.open(content_file), caption="Content Image", width=300)
    st.image(Image.open(style_file), caption="Style Image", width=300)

    if st.button("✨ Stylize Now"):
        st.info("Running Neural Style Transfer...")

        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        content_layers = ['block5_conv2']
        extractor = StyleContentModel(style_layers, content_layers)

        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        image = tf.Variable(content_image)
        optimizer = tf.optimizers.Adam(learning_rate=0.02)

        for i in range(20):  # More iterations → better output
            train_step(image, extractor, style_targets, content_targets, optimizer)

        output_image = tensor_to_image(image)

        st.image(output_image, caption="🖼️ Stylized Output", use_column_width=True)
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button("📥 Download Stylized Image", byte_im, file_name="stylized_output.jpg", mime="image/jpeg")
