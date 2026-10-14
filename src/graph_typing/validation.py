"""Parser extracted from the immutable inference runner; duplicate keys use last value."""
import json
import re

def parse_response(text, condition, stopped=True):
    if not stopped:
        return 'incomplete', [], ''
    try:
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
        obj = json.loads(clean)
        pred, explanation = obj['types'], obj.get('explanation', '')
        if not isinstance(pred, list) or not pred or any(not isinstance(x, str) or not x.strip() for x in pred):
            raise ValueError('Invalid types')
        if not isinstance(explanation, str) or ('cot' in condition and not explanation.strip()):
            raise ValueError('Missing explanation')
        return 'completed', pred, explanation
    except (ValueError, KeyError, TypeError):
        return 'parse_error', [], ''
