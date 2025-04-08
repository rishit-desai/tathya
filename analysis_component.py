import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
)
import json

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


def main(text: str):
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
                "stance": str(pred_labels_stance[i]),
                "group_appeals": str(pred_labels_group_appeals[i]),
                "endorsement": str(pred_labels_endorsement[i]),
                "hyperbole": str(pred_labels_hyperbole[i]),
                "sentiment": str(pred_labels_sentiment[i]),
                "vagueness": str(pred_labels_vagueness[i]),
                "readability": str(pred_labels_readability[i]),
            }

        json_data = {
            "text": text,
            "tokens": tokens_dict,
        }
        json_file = "output/output.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"JSON file saved to {json_file}")

main("I hate you")
