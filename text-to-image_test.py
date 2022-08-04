### Testing API

import requests

BASE = 'http://127.0.0.1:3008/text-to-image'
TEST_CASE = {
    'text_prompt': 'velcro beer',
    'model': 'MINI',
    'preds_per_prompt': 1
    }

requests.post(BASE, json=TEST_CASE)


