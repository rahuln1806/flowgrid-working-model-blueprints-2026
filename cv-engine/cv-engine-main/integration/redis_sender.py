import redis
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def send(data):
    r.set("cv:data", json.dumps(data))