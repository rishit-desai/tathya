from flask import Flask, request, jsonify
from flask_cors import CORS
import time  # For simulating processing time
import utils
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests from your React app
def create_model(tokenizer_name, model_name, shouldUseSequenceClassification=False):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, return_offsets_mapping=True
    )  # Or any other suitable model

    if shouldUseSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)  # Or any other suitable model
    return tokenizer, model

stance_tokenizer, stance_model = create_model(
        "bucketresearch/politicalBiasBERT", "bucketresearch/politicalBiasBERT"
    )
group_appeals_tokenizer, group_appeals_model = create_model(
    "rwillh11/mDeBERTa-group-appeals-detection",
    "rwillh11/mDeBERTa-group-appeals-detection",
)
endoresement_tokenizer, endoresement_model = create_model(
    "morenolq/spotify-podcast-advertising-classification",
    "morenolq/spotify-podcast-advertising-classification",
)
hyperbole_tokenizer, hyperbole_model = create_model(
    "Tuiaia/bert-base-multilingual-cased-impact-intensity",
    "Tuiaia/bert-base-multilingual-cased-impact-intensity",
)
emotion_tokenizer, emotion_model = create_model(
    "bhadresh-savani/bert-base-uncased-emotion",
    "bhadresh-savani/bert-base-uncased-emotion",
)
vagueness_tokenizer, vagueness_model = create_model(
    "Jaki01/vagueness-detection-large", "Jaki01/vagueness-detection-large"
)
readability_tokenizer, readability_model = create_model(
    "tareknaous/readabert-en", "tareknaous/readabert-en"
)
models = [
    {
        "name": "Stance",
        "tokenizer": stance_tokenizer,
        "model": stance_model,
        "num_labels": 3,
        "classes": {
            0: "left",
            1: "center",
            2: "right",
        }
    },
    {
        "name": "Group Appeals",
        "tokenizer": group_appeals_tokenizer,
        "model": group_appeals_model,
        "num_labels": 2,
    },
    {
        "name": "Endoresement",
        "tokenizer": endoresement_tokenizer,
        "model": endoresement_model,
        "num_labels": 2,
    },
    {
        "name": "Hyperbole",
        "tokenizer": hyperbole_tokenizer,
        "model": hyperbole_model,
        "num_labels": 2,
    },
    {
        "name": "Sentiment Analysis",
        "tokenizer": emotion_tokenizer,
        "model": emotion_model,
        "num_labels": 5,
        "classes": {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }
    },
    {
        "name": "Vagueness",
        "tokenizer": vagueness_tokenizer,
        "model": vagueness_model,
        "num_labels": 2,
    },
    {
        "name": "Readability",
        "tokenizer": readability_tokenizer,
        "model": readability_model,
        "num_labels": 5,
    }
    ]
pipe = pipeline("text-classification", 
                    model="q3fer/distilbert-base-fallacy-classification", 
                    tokenizer="q3fer/distilbert-base-fallacy-classification",
                    )

# In a real application, you would replace this with your actual NLP logic
def process_text(text):
    """
    Simulates NLP processing on the input text.  Replace this with your actual NLP code.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """
    
    processed_text = utils.unified_analysis(text,models,pipe=pipe)  # Call the utility function to process the text

    return processed_text


@app.route("/process_text", methods=["POST"])
def process_text_endpoint():
    """
    Endpoint to receive text from the frontend, process it, and return the result.
    """
    try:
        data = request.get_json()  # Get the JSON data from the request
        if not data or "paragraph" not in data:
            return (
                jsonify(
                    {
                        "error": 'Invalid input.  Expected a JSON object with a "paragraph" key.'
                    }
                ),
                400,
            )

        input_text = data["paragraph"]
        if not isinstance(input_text, str):
            return jsonify({"error": 'Input "paragraph" must be a string.'}), 400

        if not input_text.strip():
            return jsonify({"error": 'Input "paragraph" cannot be empty.'}), 400

        # Process the text using the simulated NLP function
        processed_text = process_text(input_text)
        return jsonify({"result": processed_text}), 200  # Return the result as JSON

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Good practice to log the error
        return (
            jsonify({"error": error_message}),
            500,
        )  # Return a JSON error response with a 500 status code


if __name__ == "__main__":
    app.run(debug=True)  # Start the Flask development server


