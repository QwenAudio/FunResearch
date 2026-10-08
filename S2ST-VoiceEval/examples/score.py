"""Run with your recordings: python examples/score.py MODEL SOURCE CANDIDATE..."""
import argparse
import json

from s2st_voiceeval import VoiceEvaluator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model")
parser.add_argument("source")
parser.add_argument("candidates", nargs="+")
args = parser.parse_args()

evaluator = VoiceEvaluator(args.model)
print(json.dumps(evaluator.rank(args.source, args.candidates), indent=2))
