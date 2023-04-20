import streamlit as st
from diffusers import DiffusionPipeline


@st.cache_resource
def load_model():
    return DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", safety_checker=lambda images, clip_input: (images, False))


st.title("Stable Diffusion using Streamlit")

with st.sidebar.expander("Prompt"):
    prompt = st.sidebar.text_area("Positive")
    negative_prompt = st.sidebar.text_area("Negative")

diffusion_step = st.sidebar.slider("Step", min_value=0, max_value=150)

model = load_model()


def generate():
    if prompt and diffusion_step:
        image = model(
            prompt=prompt, num_inference_steps=diffusion_step, negative_prompt=negative_prompt).images[0]
        st.image(
            image, f"This is your prompt: {prompt} Step: {diffusion_step}")
    else:
        st.warning("Please enter text for generation.")


st.sidebar.button("Generate", on_click=generate)
