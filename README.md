
# Instruct Pix2Pix

Ready to deploy inference endpoint for pix2pix. 

# Disclaimer

This repo uses the HuggingFace diffusers' implementation of Tim Brooks et al. Instruct Pix2pix model - https://www.timothybrooks.com/instruct-pix2pix


# How to run the service
2 options:

Without docker
```
pip install -r requirements.txt
python3 server.py
```

Or with docker

```
docker build -t pix2pix .
docker run -p 8000:8000 --gpus=all pix2pix
```

Then test it: 

Install dev dependies

```
pip install python-dotenv banana_dev
```

Write a test_name in test.py
```
python3 test.py
```

Output will be under data/output/<test_name>

For debugging
add `debug=True, auto_reload=True` in server.run() in /server.py

Note: Sanic optimisations fail at the moment so always keep debug=True



# How to interact with the server

## Model Inputs

The model accepts the following inputs:

* `prompt` (str, required)
* `image` (base64 str, required) - A base64 string of the image (data:image/type;base64,.... also accepeted) should be 512x512 or another standard Stable Diffusion 1.5 resolution for best results
* `seed` (int, optional, defaults to 42)
* `text_cfg_scale` (float, optional, default 7)
* `image_cfg_scale` (float, optional, default 1.5)
* `steps` (int, optional, default to 20)
* `randomize_cfg` (boolean, optional, default False)
* `randomize_seed` (boolean, optional, default True)
* `image_cfg_scale` (float, optional, default 1.5)

Additional parameters:
* `test_mode` (boolean, optional, default False)
* `toDataUrl` (boolean, optional, default False) - if you want output "data:image/type;base64,...."


Not implemented
* `negative_prompt`
* `num_images_per_prompt`



## Model Output

The model outputs:

A list of image objects where each has the following properties:
* `image` (base64 str) - base64 or base64 with data_url prefix if specified
* `seed` (int)
* `text_cfg_scale` (float)
* `image_cfg_scale` (float)
* `steps` (int)



## Example

- Checkout the test.py for an example


## Test the deployed service on Banana (with moderation) (python-version)
You need api & model keys in your .env
```
API_KEY=<banana-api-key>
MODEL_KEY=<banana-model-key>
```

Write a test_name in test_banana.py
```
python3 test_banana.py
```

Output will be under data/output/<test_name>


## Test the deployed service on Banana (with moderation) (python-version)
You need api & model keys in your .env
```
API_KEY=<banana-api-key>
MODEL_KEY=<banana-model-key>
```

```
python3 test_banana.py
```

## Test the deployed service on Banana (with moderation) (python-version)

```
const banana = require("@banana-dev/banana-dev")
results = await banana.run(apiKey, modelKey, modelInputs)

```




# Examples of generated images

Venus de Milo             |  Turn her into a cyborg
:-------------------------:|:-------------------------:
![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/input/venus-of-milo-512.jpg)  |  ![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/output/venus-of-milo-512.jpeg) 

<br>

Elon            |  Turn him into a cyborg
:-------------------------:|:-------------------------:
![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/input/elon-512.jpg) |  ![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/output/elon-2-512.jpeg)

<br>

# Helpful Links

Learn more about Instruct Pix2Pix here - https://www.timothybrooks.com/instruct-pix2pix

And Hugging Face support there - https://huggingface.co/timbrooks/instruct-pix2pix

<br>
