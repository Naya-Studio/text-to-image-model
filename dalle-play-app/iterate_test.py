
import requests

BASE = 'http://127.0.0.1:8008/iterate'
TEST_CASE = {
    'image_path': 'outputs/2022-08-30_22-59-01_cartoon\ dog\ on\ the\ moon/0.jpeg',
    'num_iterations': 8
    }

requests.post(BASE, json=TEST_CASE)
