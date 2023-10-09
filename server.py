# -*- coding: utf-8 -*-

import os
import logging
import argparse
from flask import Flask, request, jsonify
from intent_classifier import IntentClassifier

app = Flask(__name__)
model = IntentClassifier()

logging.basicConfig(level=logging.INFO, filename='classifier.log', filemode='a')
logger = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@app.route('/ready')
def ready():
    if model.is_ready():
        return 'OK', 200
    else:
        return 'Not ready', 423


@app.route('/intent', methods=['POST'])
def intent():
    try:
        # Check if the request has a JSON body
        if not request.json:
            return jsonify({"label": "BODY_MISSING", "message": "Request doesn't have a body."}), 400

        # Check if the 'text' field is present in the request JSON
        if 'text' not in request.json:
            return jsonify({"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}), 400

        # Check if the 'text' field is a string
        if not isinstance(request.json['text'], str):
            return jsonify({"label": "INVALID_TYPE", "message": "\"text\" is not a string."}), 400

        # Check if the 'text' field is empty
        if not request.json['text'].strip():
            return jsonify({"label": "TEXT_EMPTY", "message": "\"text\" is empty."}), 400

        # Perform intent classification (replace this with your actual intent classification logic)
        intents = model.classify_intent_function(request.json['text'])

        # Create a response with the top 3 intent predictions
        response = {"intents": intents[:3]}

        return jsonify(response)

    except Exception as e:
        logger.exception('Internal error: %s', str(e))
        # Handle internal errors
        return jsonify({"label": "INTERNAL_ERROR", "message": str(e)}), 500


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
    args = arg_parser.parse_args()
    model.load(args.model)
    logger.info('Server is starting on port %s', args.port)
    app.run(port=args.port)


if __name__ == '__main__':
    main()

#curl http://localhost:8080/ready
#curl -X POST -H "Content-Type: application/json" -d '{"text": "when is the next flight to new york"}' http://localhost:8080/intent