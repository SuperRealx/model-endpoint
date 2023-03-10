# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

from sanic import Sanic, response
import subprocess
import app as user_src

import traceback
import sys

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
server = Sanic("DreamRoom")

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    print("received request")
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/', methods=["POST"]) 
def inference(request):
    print("received request")
    try:
        try:
            model_inputs = response.json.loads(request.json)
        except:
            model_inputs = request.json

        output = user_src.inference(model_inputs)

        return response.json(output, status=200)
    
    except Exception as e:
        exc_info = sys.exc_info()
        err=''.join(traceback.format_exception(*exc_info))
        return response.json({'message': err}, status=500)

    


if __name__ == '__main__':
    print(f"Running python: {sys.version_info[:] = }")
    server.run(host='0.0.0.0', port=8000, workers=1,  debug=True) # auto_reload=True