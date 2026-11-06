# factcheck.py
import re
import torch
from typing import List
import numpy as np
import spacy
import gc


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert logits into probabilities
            probabilities = torch.softmax(logits, dim=-1)[0]

            # Index 0 represents entailment
            entailment_score = probabilities[0].item()
            neutral_score = probabilities[1].item()
            contradiction_score = probabilities[2].item()

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probabilities
        gc.collect()

        return entailment_score, neutral_score, contradiction_score


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold=0.62):
        self.threshold = threshold

        self.stop_words = {
            "a", "an", "the", "is", "was", "were", "are",
            "am", "be", "been", "being", "in", "on", "at",
            "of", "to", "for", "and", "or", "but", "by",
            "with", "from", "as", "it", "its", "this",
            "that", "he", "she", "they", "his", "her"
        }

    def tokenize(self, text):
        words = re.findall(r"\b[a-z0-9]+\b", text.lower())

        return[
            word for word in words
            if word not in self.stop_words 
        ]
    
    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_words = set(self.tokenize(fact))

        if not fact_words:
            return "NS"

        best_score = 0.0

        for passage in passages:

            passage_words = set(
                self.tokenize(passage["text"])
            )

            overlap = fact_words.intersection(passage_words)

            score = len(overlap) / len(fact_words)

            best_score = max(best_score, score)

        if best_score >= self.threshold:
            return "S"

        return "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, threshold=0.2):
        self.ent_model = ent_model
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:

        if not passages:
            return "NS"

        best_score = -1.0

        for passage in passages:
            premise = passage["text"]
            hypothesis = fact
            entailment, neutral, contradiction = self.ent_model.check_entailment(premise, hypothesis)

            # Compare entailment against the strongest alternative
            score = entailment

            best_score = max(best_score, score)

        if best_score >= self.threshold:
            return "S"

        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
