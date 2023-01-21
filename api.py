from dataclasses import asdict

from flask import Flask, request, jsonify

from search import default_init

app = Flask(__name__)
index = default_init()


@app.route('/', methods=["POST"])
def remove_background():
    in_json = request.json
    text = in_json["text"]
    n = in_json.get("n", 1)
    output = index.query(text, n)
    out_json = jsonify([
        {key: value for key, value in asdict(el).items()}
        for el in output
    ])
    return out_json

