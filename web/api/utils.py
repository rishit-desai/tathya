import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
from transformers import (
    BertTokenizer,
    BertModel,
)
import json

from google import genai
from google.genai import types
import base64
import time

import re
import torch.nn as nn


model_1_path = "group_1_model.pth"
model_2_path = "group_2_model.pth"
model_3_path = "group_3_model.pth"


class MultiLabelBERT_Political(torch.nn.Module):
    """BERT model for multi-label token classification."""

    def __init__(
        self,
        bert_model_name,
        num_stance_labels,
        num_group_appeals_labels,
        num_endorsement_labels,
    ):
        super(MultiLabelBERT_Political, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.stance_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_stance_labels
        )
        self.group_appeals_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_group_appeals_labels
        )
        self.endorsement_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_endorsement_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, max_len, hidden_size)

        stance_logits = self.stance_classifier(sequence_output)
        group_appeals_logits = self.group_appeals_classifier(sequence_output)
        endorsement_logits = self.endorsement_classifier(sequence_output)

        return stance_logits, group_appeals_logits, endorsement_logits


class MultiLabelBERT_Sentiment(torch.nn.Module):
    """BERT model for multi-label token classification."""

    def __init__(self, bert_model_name, num_hyperbol_labels, num_sentiment_labels):
        super(MultiLabelBERT_Sentiment, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hyperbole_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_hyperbol_labels
        )
        self.sentiment_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_sentiment_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, max_len, hidden_size)

        hyperbole_logits = self.hyperbole_classifier(sequence_output)
        sentiment_logits = self.sentiment_classifier(sequence_output)

        return hyperbole_logits, sentiment_logits


class MultiLabelBERT_Readability(torch.nn.Module):
    """BERT model for multi-label token classification."""

    def __init__(self, bert_model_name, num_vagueness_labels, num_readability_labels):
        super(MultiLabelBERT_Readability, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.vagueness_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_vagueness_labels
        )
        self.readability_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_readability_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, max_len, hidden_size)

        vagueness_logits = self.vagueness_classifier(sequence_output)
        readability_logits = self.readability_classifier(sequence_output)

        return vagueness_logits, readability_logits





def token_classification(text, num_labels, tokenizer, model):
    """
    Performs token classification on the given text using a pre-trained transformer model.

    Args:
        text (str): The input text.
        num_labels (int): The number of classes for token classification.

    Returns:
        tuple: A tuple containing:
            - predictions (torch.Tensor): The predicted class labels for each token.
            - offset_mapping (list): The mapping between tokens and original text spans.
    """

    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.offset_mapping  # Store the offset mapping

    # split the total tokens into batches of 512 tokens
    batch_size = 512
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    token_type_ids = (
        inputs.token_type_ids if hasattr(inputs, "token_type_ids") else None
    )
    num_batches = input_ids.size(1) // batch_size
    if input_ids.size(1) % batch_size != 0:
        num_batches += 1
    last_hidden_states = []
    model.eval()  # Set the model to evaluation mode
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, input_ids.size(1))
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask[:, start:end],
                token_type_ids=token_type_ids[:, start:end],
            )  # Corrected: No offset_mapping
        last_hidden_states.append(outputs.last_hidden_state)
    last_hidden_states = torch.cat(last_hidden_states, dim=1)

    classifier = nn.Linear(model.config.hidden_size, num_labels)
    logits = classifier(last_hidden_states)

    predictions = torch.argmax(logits, dim=-1)

    decoded_predictions = []
    for i, token_preds in enumerate(
        predictions
    ):  # Iterate over the batch (only 1 example here)
        token_decoded_preds = []
        for j, pred in enumerate(token_preds):
            start, end = offset_mapping[0][j]
            original_span = text[start:end]
            token_decoded_preds.append(
                (original_span, pred.item())
            )  # pred.item() to get the int value

        decoded_predictions.append(token_decoded_preds)

    return decoded_predictions[0]


def get_char_offsets(text, tokens):
    """
    Returns a list of character offsets for each token in the text.

    Args:
        text (str): The input text.
        tokens (list): A list of tokens.

    Returns:
        list: A list of tuples, where each tuple contains the start and end character
            offsets for a token.
    """
    char_offsets = []
    start = 0
    for token in tokens:
        start = text.find(token, start)
        end = start + len(token)
        char_offsets.append((start, end))
        start = end
    return char_offsets


def align_predictions_word_level(text, model_outputs):
    """
    Aligns token-level predictions from multiple models to word-level representations.
    Performs majority vote aggregation for each model's predictions within each word.

    Args:
        text (str): The original input text.
        model_outputs (list): A list of dictionaries, where each dictionary represents
            the output of a model. Each dictionary should have the following keys:
            - "tokens": A list of tokens.
            - "predictions": A list of corresponding predictions.
            - "get_char_offsets": A function that takes the text and tokens and returns
              a list of tuples, where each tuple contains the start and end character
              offsets for a token.

    Returns:
        dict: A dictionary where keys are words, and values are dictionaries containing
            the majority vote prediction for each model within that word.
    """

    words = re.findall(r"\b\w+\b", text)
    aligned_data = {}

    for word in words:
        word_start = text.find(word)
        word_end = word_start + len(word)

        aligned_data[word] = {}

        for model_output in model_outputs:
            model_name = f"model_{model_output['name']}"
            model_predictions = []
            char_offsets = model_output["get_char_offsets"](
                text, model_output["tokens"]
            )

            for i, (start, end) in enumerate(char_offsets):
                if start >= word_start and end <= word_end:
                    model_predictions.append(model_output["predictions"][i])

            if model_predictions:
                counts = {}
                for prediction in model_predictions:
                    counts[prediction] = counts.get(prediction, 0) + 1
                majority_prediction = max(counts, key=counts.get)
                aligned_data[word][model_name] = majority_prediction

    return aligned_data


def analysis_classic(text: str,models,pipe):
    row = {}
    row["index"] = 0
    row["tokens"] = {}
    max_index = 0
    model_outputs = []
    for model in models:
        decoded_predictions = []

        decoded_predictions = token_classification(
            text, model["num_labels"], model["tokenizer"], model["model"]
        )

        model_outputs.append(
            {
                "tokens": [
                    token_text for token_text, prediction in decoded_predictions
                ],
                "predictions": [
                    prediction for token_text, prediction in decoded_predictions
                ],
                "get_char_offsets": get_char_offsets,
                "name": model["name"],
            }
        )

    row["fallacy"] = pipe(text[:1024])[0]
    row["tokens"] = align_predictions_word_level(text, model_outputs)

    # convert the dictionary to a JSON string
    json_string = json.dumps(row, indent=4)
    return json_string


def analysis(text: str):
    # Load the model and tokenizer
    model_1 = MultiLabelBERT_Political(
        bert_model_name="bert-base-uncased",
        num_stance_labels=3,
        num_group_appeals_labels=2,
        num_endorsement_labels=2,
    )
    model_1.load_state_dict(torch.load(model_1_path))
    model_1.eval()
    model_2 = MultiLabelBERT_Sentiment(
        bert_model_name="bert-base-uncased",
        num_hyperbol_labels=2,
        num_sentiment_labels=6,
    )
    model_2.load_state_dict(torch.load(model_2_path))
    model_2.eval()
    model_3 = MultiLabelBERT_Readability(
        bert_model_name="bert-base-uncased",
        num_vagueness_labels=2,
        num_readability_labels=6,
    )
    model_3.load_state_dict(torch.load(model_3_path))
    model_3.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1.to(device)
    model_2.to(device)
    model_3.to(device)

    # Load the data
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        stance_logits, group_appeals_logits, endorsement_logits = model_1(
            inputs["input_ids"], inputs["attention_mask"]
        )
        hyperbole_logits, sentiment_logits = model_2(
            inputs["input_ids"], inputs["attention_mask"]
        )
        vagueness_logits, readability_logits = model_3(
            inputs["input_ids"], inputs["attention_mask"]
        )

        pred_labels_stance = torch.argmax(stance_logits, dim=2).squeeze().cpu().numpy()
        pred_labels_group_appeals = (
            torch.argmax(group_appeals_logits, dim=2).squeeze().cpu().numpy()
        )
        pred_labels_endorsement = (
            torch.argmax(endorsement_logits, dim=2).squeeze().cpu().numpy()
        )
        pred_labels_hyperbole = (
            torch.argmax(hyperbole_logits, dim=2).squeeze().cpu().numpy()
        )
        pred_labels_sentiment = (
            torch.argmax(sentiment_logits, dim=2).squeeze().cpu().numpy()
        )
        pred_labels_vagueness = (
            torch.argmax(vagueness_logits, dim=2).squeeze().cpu().numpy()
        )
        pred_labels_readability = (
            torch.argmax(readability_logits, dim=2).squeeze().cpu().numpy()
        )

        # get tokens
        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().cpu().numpy()
        )
        tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]]
        tokens = tokens[: len(pred_labels_stance)]
        pred_labels_stance = pred_labels_stance[: len(tokens)]
        pred_labels_group_appeals = pred_labels_group_appeals[: len(tokens)]
        pred_labels_endorsement = pred_labels_endorsement[: len(tokens)]
        pred_labels_hyperbole = pred_labels_hyperbole[: len(tokens)]
        pred_labels_sentiment = pred_labels_sentiment[: len(tokens)]
        pred_labels_vagueness = pred_labels_vagueness[: len(tokens)]
        pred_labels_readability = pred_labels_readability[: len(tokens)]

        # # get labels
        # stance_labels = ["Left", "Center", "Right"]
        # group_appeals_labels = ["No Group Appeal", "Group Appeal"]
        # endorsement_labels = ["No Endorsement", "Endorsement"]
        # hyperbole_labels = ["No Hyperbole", "Hyperbole"]
        # sentiment_labels = [
        #     "Joy",
        #     "Sadness",
        #     "Anger",
        #     "Fear",
        #     "Love",
        #     "Surprise",
        # ]
        # vagueness_labels = ["No Vagueness", "Vagueness"]
        # readability_labels = [
        #     "Extremely Hard",
        #     "Hard",
        #     "Normal",
        #     "Easy",
        #     "Very Easy",
        #     "Extremely Easy",
        # ]

        # # get predictions
        # pred_stance = [stance_labels[label] for label in pred_labels_stance]
        # pred_group_appeals = [
        #     group_appeals_labels[label] for label in pred_labels_group_appeals
        # ]
        # pred_endorsement = [
        #     endorsement_labels[label] for label in pred_labels_endorsement
        # ]
        # pred_hyperbole = [hyperbole_labels[label] for label in pred_labels_hyperbole]
        # pred_sentiment = [
        #     sentiment_labels[label] for label in pred_labels_sentiment
        # ]
        # pred_vagueness = [vagueness_labels[label] for label in pred_labels_vagueness]
        # pred_readability = [
        #     readability_labels[label] for label in pred_labels_readability
        # ]

        # create json
        tokens_dict = {}
        for i, token in enumerate(tokens):
            tokens_dict[token] = {
                "token": token,
                "1": str(pred_labels_stance[i]),
                "2": str(pred_labels_group_appeals[i]),
                "3": str(pred_labels_endorsement[i]),
                "4": str(pred_labels_hyperbole[i]),
                "5": str(pred_labels_sentiment[i]),
                "6": str(pred_labels_vagueness[i]),
                "7": str(pred_labels_readability[i]),
            }

        json_data = {
            "text": text,
            "tokens": tokens_dict,
        }
        json_file = "output/output.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"JSON file saved to {json_file}")

def generative(json_string, prompt):
    # Load the JSON string into a Python dictionary
    data = json.loads(json_string)

    # Extract the text and tokens from the dictionary
    text = data["text"]
    tokens = data["tokens"]

    # Generate the output based on the prompt
    output = f"Text: {text}\n\n"
    for token, attributes in tokens.items():
        output += f"Token: {token}\n"
        for key, value in attributes.items():
            output += f"{key.capitalize()}: {value}\n"
        output += "\n"

    # Add the prompt to the output
    output += f"Prompt: {prompt}\n"

    return output

def generate(article_analysis):
    client = genai.Client(
        vertexai=True,
        project="tathya-456709",
        location="us-central1",
    )

    static_prompt = """Analyze the provided article focusing on the reporting of a specific controversial event or situation described within it,
where "model_1","model_2","model_3","model_4","model_5","model_6","model_7" refers to Political Stance (0-left, 1-center, 2-right), group appeal, endorsement, hyperbole, sentiment (0 - sadness, 1 -joy, 2 - love, 3 - anger, 4 - fear, 5 - surprise,), vagueness and readability respectively
Pay close attention to the language used to present:
The background and context leading up to the event.
The actions and perspectives of the key individuals or groups involved.
Any disputes, disagreements, or conflicting accounts surrounding the event.
The consequences or outcomes resulting from the event.
For each of these points, identify potential areas of bias, lack of neutrality, insufficient context, or missing information. Provide specific examples from the text (quotations or paraphrases) to support your analysis. Strive for objectivity and avoid expressing personal opinions or assumptions about the events or individuals involved. Your analysis should focus on the reporting of the event, not the event itself."
General Instructions for Use with "Token Annotations":
Identify the Central Event: First, clearly identify the specific controversial event or situation that the article focuses on. This will guide your analysis.
Extract Key Terms and Phrases: As you read, highlight or note down specific words and phrases (tokens) that are relevant to the four points in the prompt. These might include:
Terms describing individuals or groups
Verbs describing actions
Adjectives or adverbs conveying judgment
Phrases expressing opinions or beliefs
Words indicating cause and effect
Analyze Token Connotations: For each token, analyze its potential connotations:
Sentiment: Does it express positive, negative, or neutral sentiment?
Bias: Does it suggest a particular viewpoint or prejudice?
Precision: Is it specific and factual, or vague and emotive?
Examine Framing and Language Choice: Consider how the tokens are used to frame the event and portray the involved parties.
Is the language balanced and fair, or does it favor one side?
Are there loaded terms or euphemisms that influence the reader's perception?
Are certain voices or perspectives marginalized or omitted?
Contextualize Token Usage: Analyze how the context influences the meaning and impact of the tokens.
A seemingly neutral word can become biased in a specific context.
Pay attention to how tokens are used in relation to different individuals or groups.
Evaluate Attribution and Evidence: Assess how the article attributes statements, claims, and opinions.
Are sources clearly identified and credible?
Is there sufficient evidence to support the claims being made?
Are different perspectives given fair representation?
Synthesize and Conclude: Combine your token-level analysis to draw overall conclusions about the article's reporting of the central event.
Is the reporting objective and balanced?
Are there significant biases or omissions?
Does the language used contribute to a clear and accurate understanding of the event?
Do not mention that you are a chatbot and answer without refering to yourself or your act of answering, directly start answering
"""

    unified_prompt = types.Part.from_text(text=str(article_analysis+static_prompt))

    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(role="user", parts=[unified_prompt]),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
    )

    final_string = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            final_string += chunk.text

    return final_string

def unified_analysis(text: str,models,pipe):
    json_string = analysis_classic(text,models=models,pipe=pipe)
    article_analysis = generate(json_string)
    return article_analysis
