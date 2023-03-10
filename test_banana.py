# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``
import os
import base64
import requests
import banana_dev as banana  
from utils import stringToPil
from dotenv import load_dotenv


# params
test_name = "van-gogh-girl-selfie" #"van-gogh-macron" #"van-gogh-cool-guy-selfie"

test_img_path = f"data/input/girl-selfie.jpg"
prompt = "make it a van gogh painting"
num_inference_steps=30
image_guidance_scale=1.4
prompt_guidance_scale=7
num_images_per_prompt= 1
test_mode=False


load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL_KEY = os.getenv("MODEL_KEY")

def make_test_example():
    # convert img to base64
    with open(test_img_path, "rb") as image2string:
        base64_bytes  = base64.b64encode(image2string.read())

    # pass it as string for json
    base64_string = base64_bytes.decode('utf-8')

    model_inputs = {'prompt': prompt,
                    'image': base64_string,
                    'steps': num_inference_steps,
                    'image_cfg_scale': image_guidance_scale,
                    'text_cfg_scale': prompt_guidance_scale,
                    'num_images_per_prompt': num_images_per_prompt,
                    'test_mode': test_mode
                    }
    return model_inputs

def test_inference():
    
    model_inputs = make_test_example()

    response = banana.run(API_KEY, MODEL_KEY, model_inputs)

    print("Server Inference with: ", response.status_code)

    # save responnse
    output = response.json()

    try:
        for idx, image in enumerate(output): 
            print("Saving images")

            txt_cfg = image["text_cfg_scale"]
            img_cfg = image["image_cfg_scale"]
            steps = image["steps"]

            img_path = f"data/output/{test_name}-{txt_cfg}-{img_cfg}-{steps}-{idx}.jpeg"
            print("Saving: ", img_path)

            img = stringToPil(image['image'])
            img.save(img_path)
    except Exception as e:
        print(output)
        print(e)


if __name__ == "__main__":

    try:
        print("Testing inference")
        test_inference()
    except Exception as e:
        print(e)