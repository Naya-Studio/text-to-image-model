
import requests

BASE = 'http://127.0.0.1:8008/dalle'
TEST_CASE = {
    'text': 'cartoon dog on the moon',
    'num_images': 8
    }

requests.post(BASE, json=TEST_CASE)
