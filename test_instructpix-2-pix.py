if __name__ == "__main__":
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    from diffusers.utils import load_image  

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    grayscale_image_path = "/accounts/grad/phudish_p/CS280A_final_project/original_image_1.png"
    original_image = load_image(grayscale_image_path)
    original_image = original_image.convert("RGB")
    original_image.save("pix2pix_original_image.png")

    prompt = "colorize the image"
    images = pipe(prompt, image=original_image, num_inference_steps=10, image_guidance_scale=1).images
    images[0].save("pix2pix.png")
