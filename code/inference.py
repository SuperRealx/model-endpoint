import math
import json
import random
import logging
from PIL import Image, ImageOps

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler


# This code will be loaded on each worker separately..
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    device = _get_device()
    logger.info(">>> Device is '%s'.." % device)
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None)
    model.to(device)
    print(type(model))
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    logger.info(">>> Model loaded!..")
    return model


# Inference is ran for every server call
# Reference preloaded global pipeline here. 
def transform_fn(
    model, 
    request_body, 
    content_type, 
    accept
)->list:
    """ Runs image transformations. 

    Args: 
        model_inputs (dict): a key value store of the inputs for pix2pix
    Returns:
        list of generated images objects
    """
    model_inputs = request_body["model_inputs"]
    # Parse pipeline arguments
    # Official model inputs
    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('steps', 20)
    image_cfg_scale=model_inputs.get('image_cfg_scale', 1.5)
    text_cfg_scale=model_inputs.get('text_cfg_scale', 7)
    seed=model_inputs.get('seed', 42)
    randomize_cfg=model_inputs.get('randomize_cfg', False)
    randomize_seed=model_inputs.get('randomize_seed', True)
    num_images_per_prompt=model_inputs.get('num_images', 1)
        
    # Custom
    just_test_img=model_inputs.get('test_mode', False)
    toDataUrl=model_inputs.get('toDataUrl', False)

    # decode image
    base64_string = model_inputs.get('image')
    image = _stringToPil(base64_string)

    if prompt == None:
        return {'message': "No prompt provided"}

    results = []
    if just_test_img:
        # just test sending & getting back images
        results.append({
            "seed": seed, 
            "text_cfg_scale":text_cfg_scale, 
            "image_cfg_scale":image_cfg_scale, 
            "steps": steps,
            'image': _pilToString(image, dataUrl=toDataUrl)
            })

    else:
        # preprocessing step
        preprocessed_img = _preprocess(image)

        # generate different samples
        sample_variants = [
            (image_cfg_scale, text_cfg_scale),
            (round(image_cfg_scale-.15, 3), text_cfg_scale),
            (round(image_cfg_scale-.3, 3), text_cfg_scale),
            (round(image_cfg_scale+.15, 3), text_cfg_scale),
            (image_cfg_scale, text_cfg_scale+2)

        ]
        
        for i, (img_cfg, text_cfg) in enumerate(sample_variants):
            logger.info("Running for sample: %s", i)
            i+=1
            # Run the model
            results.append(
                _generate(
                    model,
                    prompt=prompt,
                    input_image=preprocessed_img,
                    steps=steps,
                    randomize_seed=randomize_seed,
                    seed=seed,
                    randomize_cfg=randomize_cfg,
                    text_cfg_scale=text_cfg,
                    image_cfg_scale=img_cfg,
                    num_images_per_prompt=num_images_per_prompt
                    )
                )
    return json.dumps(results)


def _get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def _preprocess(
    image: str
)->Image.Image:
    """ Resize the image if needed. 
        Args:
            image_64_string (str) 
        Returns: 
            Image.Image
    """
    width, height = image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

    return input_image
    
    
def _generate(
    model,
    prompt: str,
    input_image: Image.Image,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    num_images_per_prompt: int
)->list:
    """Runs the model to generate the edited images. 

    Args:
        prompt (str)
        input_image (Image.Image)
        steps (int)
        randomize_seed (bool)
        seed (int)
        randomize_cfg (bool)
        text_cfg_scale (float)
        image_cfg_scale (float)
        num_images_per_prompt (int)

    Returns:
        dict: generated images objects
    """
    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    generator = torch.manual_seed(seed)

    # debugging
    print("Running model with params: ", 
        {'prompt': prompt, 
        'steps': steps, 
        'image_cfg_scale': image_cfg_scale, 
        'text_cfg_scale': text_cfg_scale,
        'image_size': input_image.size,
        "num_images_per_prompt": num_images_per_prompt})

    res = []
    i = 0
    #for (img_cfg, text_cfg) in grid_search:
    i+=1

    img = model(prompt, image=input_image, 
                num_inference_steps=steps, 
                image_guidance_scale=image_cfg_scale, 
                guidance_scale=text_cfg_scale, 
                generator=generator).images[0]
    
    return {
        'seed':seed, 
        'text_cfg_scale':text_cfg_scale, 
        'image_cfg_scale':image_cfg_scale, 
        "steps": steps,
        'image': _pilToString(img)
        }



def _stringToPil(
    img_string: str
    ):
    is_data_url = True if len(img_string.split(",")) > 1 else False
    
    if is_data_url:
        base64string = img_string.split(",")[1]
    else:
        base64string = img_string

    img = Image.open(BytesIO(base64.b64decode(base64string,
         validate=True))).convert("RGB")
    return img


def _pilToString(
    img: Image, 
    dataUrl=False
    ):
    
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()

    base64string =  base64.b64encode(im_bytes).decode('utf-8')
    if dataUrl:
        return 'data:image/jpeg;base64,' + base64string
    else:
        return base64string