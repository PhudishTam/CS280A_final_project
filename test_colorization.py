from huggingface_hub import login
from PIL import Image
import torch
from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import load_image



if __name__ == "__main__":
    token = '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'
    login(token=token)

    # Load and preprocess the grayscale image
    grayscale_image_path = "/accounts/grad/phudish_p/CS280A_final_project/dataset/Gray/Apple/Apple1.jpg"
    original_image = load_image(grayscale_image_path)
    # convert the image to grayscale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_image = original_image.convert("L")
    original_image.save("original_image.png")
    stage_1 = IFImg2ImgPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    )
    stage_1.to(device)

    # Load DeepFloyd IF stage II (Super Resolution Pipeline)
    stage_2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_2.to(device)

    # Add stage 3 with safety modules from stage 1
    safety_modules = {
        "feature_extractor": stage_1.feature_extractor,
        "safety_checker": stage_1.safety_checker,
        "watermarker": stage_1.watermarker,
    }
    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        **safety_modules,
        torch_dtype=torch.float16,
    )
    stage_3.to(device)

    # Encode the prompt
    prompt = "a colorful photo of a red apple"
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

    # Generate the image using stage I
    with torch.no_grad():
        generated_image_stage_1 = stage_1(
            image=original_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=10,
            output_type="pt",
        ).images

    # Generate the image using stage II
    with torch.no_grad():
        generated_image_stage_2 = stage_2(
            image=generated_image_stage_1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            original_image=original_image,
            num_inference_steps=10,
            output_type="pt",
        ).images

    # Generate the final high-resolution image using stage III
    with torch.no_grad():
        stage_3_output = stage_3(
            prompt=prompt,
            image=generated_image_stage_2,
            generator=torch.manual_seed(1),
            noise_level=100,
        ).images

    # Save or display the final image
    stage_3_output[0].save("generated_image.png")