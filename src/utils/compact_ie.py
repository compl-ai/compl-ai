import requests
from nltk import sent_tokenize

COMPACT_IE_SERVICE_HOST = "0.0.0.0"
COMPACT_IE_SERVICE_PORT = 39881


def extract_triples_compact_ie(text: str):
    """Uses the CompactIE API to extract triples, running locally"""
    request = {"sentences": [s for s in sent_tokenize(text)]}
    url = f"http://{COMPACT_IE_SERVICE_HOST}:{COMPACT_IE_SERVICE_PORT}/api"
    result = requests.post(url, json=request).json()
    return [(a["subject"], a["relation"], a["object"]) for a in result]


def extract_triples_compact_ie_mock(text: str):
    """Mock function to extract triples from text"""
    return [("A", "B", "C")]
