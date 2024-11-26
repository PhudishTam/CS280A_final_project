
if __name__ == "__main__":
    import requests
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    img_url = "initData/MS_COCO/val_set/val2017/000000533145.jpg"
    raw_image = Image.open(img_url).convert("RGB")

    # conditional image captioning
    text = "A colorful image of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # from PIL import Image

    # # Load model directly
    # from transformers import Blip2Processor, Blip2ForConditionalGeneration


    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco").to("cuda")
    # # Load your image
    # image = Image.open("initData/MS_COCO/val_set/val2017/000000579070.jpg").convert("RGB")  # Replace with your image path

    # # Define your captions
    # captions = [
    # "A group of people sitting around a table.",
    # "A group of people gathered around the table.",
    # "A group of people sitting at a table with many small sip cups.",
    # "Five people sitting together at a table with Dixie cups.",
    # "A group of friends sitting around a wooden table together."
    # ]

    # # Combine captions into a single input prompt
    # combined_captions = " ".join(captions)
    # print(combined_captions)
    # prompt = f"Based on this image and the following captions:\n{combined_captions}\nProvide a detailed and vivid description of the image that goes beyond the captions and return one caption ? Answer:"

  
    # inputs = processor(image, prompt, return_tensors="pt").to("cuda")

    # outputs = model.generate(**inputs, max_new_tokens=100)

    # # Decode the generated caption
    # enhanced_caption = processor.decode(outputs[0], skip_special_tokens=True).strip()

    # print("Generated Visual Caption:")
    # print(enhanced_caption)
    # from transformers import FuyuProcessor, FuyuForCausalLM
    # from PIL import Image
    # import requests

    # # load model and processor
    # model_id = "adept/fuyu-8b"
    # processor = FuyuProcessor.from_pretrained(model_id)
    # model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

    # # prepare inputs for the model
    # image = Image.open("initData/MS_COCO/val_set/val2017/000000579070.jpg").convert("RGB")  # Replace with your image path

    # # Define your captions
    # captions = [
    # "A group of people sitting around a table.",
    # "A group of people gathered around the table.",
    # "A group of people sitting at a table with many small sip cups.",
    # "Five people sitting together at a table with Dixie cups.",
    # "A group of friends sitting around a wooden table together."
    # ]

    # # Combine captions into a single input prompt
    # combined_captions = " ".join(captions)
    # #print(combined_captions)
    # #text_prompt = f"Based on this image and the following captions:\n{combined_captions}\nProvide a detailed and vivid description of the image that goes beyond the captions and return one caption? Answer:"
    # text_prompt = f"Based on this image and the following captions:\n{combined_captions}\nProvide a detailed and vivid description of the image that includes specific details about colors and goes beyond the captions. Answer:"
    # inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

    # # autoregressively generate text
    # generation_output = model.generate(**inputs, max_new_tokens=100)
    # generation_text = processor.batch_decode(generation_output[:, -100:], skip_special_tokens=True)
    # print(generation_text)
    # from transformers import pipeline

    # # Initialize summarization pipeline
    # summarizer = pipeline("summarization")

    # # Captions
    # captions = [
    #     "A group of people sitting around a table.",
    #     "A group of people gathered around the table.",
    #     "A group of people sitting at a table with many small sip cups.",
    #     "Five people sitting together at a table with Dixie cups.",
    #     "A group of friends sitting around a wooden table together."
    # ]

    # # Combine captions into one text
    # input_text = " ".join(captions)

    # # Summarize the captions
    # summary = summarizer(input_text, min_length=20, max_length=30)

    # # Add the condition "A colorful image of" to the summary
    # final_caption = f"A colorful image of {summary[0]['summary_text']}"

    # print("Final Caption for Guidance:")
    # print(final_caption)

    