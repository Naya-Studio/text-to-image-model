### Testing API

import requests

BASE = 'http://127.0.0.1:3008/timing'
TEST_CASE = {}

response = requests.post(BASE, json=TEST_CASE)


