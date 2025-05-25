import torch
import diffusers
from PIL import ImageDraw, ImageFont
import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32


@st.cache_resource
def load_model():
    MODEL_NAME = os.getenv("MODEL_NAME")
    LORA_WEIGHTS = os.getenv("LORA_WEIGHTS")
    
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    )
    pipeline.to(device)
    
    pipeline.load_lora_weights(
        LORA_WEIGHTS, # Directory containing weights file
        weight_name="pytorch_lora_weights.safetensors",
    )

    return pipeline


def generate_images(prompt, pipeline, n):
    """
    Args:
        prompt: A image generation prompt, as a string.
        pipeline: A Stable Diffusion pipeline object.
        n: The number of images to create, as an integer.
    Return:
         A list of PIL Images of length n.
    """
    prompts = [prompt] * 4
    images = pipeline(prompts).images
    
    return images

def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    # Initialization
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Calculate the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position at which to draw the text to center it
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)




def generate_memes(prompt, text, pipeline, n):
    """
    Args:
        prompt: A image generation prompt, as a string.
        text: The text to superimpose on the images, as a string.
        pipeline: A Stable Diffusion pipeline object.
        n: The number of images to create, as an integer.
    Return:
         n images from the prompt, add the text to each one, and return a list of PIL Images.
    """
    images = generate_images(prompt, pipeline, n)
    for image in images:
        add_text_to_image(image, text)

    return images



def main():
    st.sidebar.header("Input Options")
    pipeline = load_model()
    
    number_of_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=10, value=3)
    prompt = st.sidebar.text_area("Text-to-Image Prompt", "My dog Maya")
    text = st.sidebar.text_area("Text to Display", "Maya!")

    st.markdown("## Diffusion Model Image Generator")

    if st.sidebar.button("Generate Images"):
        if not prompt:
            st.error("Please provide a Text-to-Image Prompt.")
        elif not text:
            st.error("Please provide Text to Display.")
        else:
            # st.write(f'{number_of_images} images with prompt "{prompt}" and text "
            # {text}".')
            with st.spinner("Generating images..."):
                images = generate_memes(prompt, text, pipeline, number_of_images)
                for img  in images:
                    st.image(img)
    
if __name__ == '__main__':
    main()







