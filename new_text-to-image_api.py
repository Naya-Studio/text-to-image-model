
### API FOR TEXT-TO-IMAGE MODEL ###

import datetime

from flask import Flask, request
import json
import os

import jax
import jax.numpy as jnp

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

from flax.jax_utils import replicate
from functools import partial

import random

from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange


port = os.environ.get('PORT', 3008)


######## UPON LOAD (1ST) ########




######## UPON REQUEST ########

app = Flask(__name__)

@app.route('/text-to-image', methods=['POST'])
def text_to_image():

    print('request made:',datetime.datetime.now(),'\n')

    INPUT = request.json
    text_prompt =str (INPUT['text_prompt'])
    model = str(INPUT['model'])
    preds_per_prompt = int(INPUT['preds_per_prompt'])


    ## DALLEE model
    if model == 'MINI':
        DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
    elif model == 'MEGA':
        DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
    else:
        print('specify dalle model or default as MINI')
    DALLE_COMMIT_ID = None


    ## Making prompts
    if type(text_prompt) == list:
        PROMPTS = text_prompt
    elif type(text_prompt) == str:
        PROMPTS = [text_prompt]


    ## VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    print('loading model and tokens:',datetime.datetime.now())
    ## Load models & tokenizer: LONG LOAD TIME ON FIRST USE
    ## Load dalle-mini
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )

    ## Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )
    print('models and tokens loaded:',datetime.datetime.now(),'\n')

    print('replicating params:',datetime.datetime.now())
    ## Model parameters are replicated on each device for faster inference
    params, vqgan_params = replicate(params), replicate(vqgan_params)
    ## Model functions are compiled and parallelized to take advantage of multiple devices
    ## model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    ## decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)
    print('params replicated:',datetime.datetime.now(),'\n')

    print('generating keys:',datetime.datetime.now())
    ## Keys are passed to the model on each device to generate unique inference per device.
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    print('keys generated:',datetime.datetime.now(),'\n')

    print('processing prompts:',datetime.datetime.now())
    ## Processing prompts
    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
    tokenized_prompts = processor(PROMPTS)
    tokenized_prompt = replicate(tokenized_prompts)
    print('prompts processed:',datetime.datetime.now(),'\n')

    print('generating images:',datetime.datetime.now())
    ## Generate images using dalle-mini model and decode them with the VQGAN.
    name = 'test'
    if type(name) != str:
        print('name of image must be string')
        return
    ## number of predictions per prompt
    n_predictions = preds_per_prompt
    ## We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    print(f"Prompts: {PROMPTS}\n")
    ## generate images
    images = []
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        ## get a new key
        print('getting new key:',datetime.datetime.now())
        key, subkey = jax.random.split(key)
        # generate images
        print('encoding images:',datetime.datetime.now())
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        print('remove bos:',datetime.datetime.now())
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        print('decoding images:',datetime.datetime.now())
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
            img.save('gen_image_'+name+'.jpg')
            
    print('images generated:',datetime.datetime.now(),'\n')

    print('request returned:',datetime.datetime.now())

    return


######## UPON LOAD (2ND) ########
    
if __name__ == "__main__":
    import tensorflow as tf
    print('api loaded:',datetime.datetime.now(),'\n')
    #app.run(host='0.0.0.0',port=port)
    app.run(host='0.0.0.0',port=port,debug=True)

























