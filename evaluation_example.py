from human_eval.evaluation import evaluate_functional_correctness
from human_eval.data import stream_jsonl
import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(ROOT, "..", "data", "example_problem.jsonl")
COMPLETIONS = os.path.join(ROOT, "..", "data", "example_samples.jsonl")
RESULTS = os.path.join(ROOT, "..", "data", "example_samples.jsonl_results.jsonl")

# results = evaluate_functional_correctness(sample_file=COMPLETIONS, k=[2], problem_file=PROBLEMS)

# msg = """This is
# a multiline
#     test message
# !!
# """

# with open('./tst_results.jsonl', "ab") as fp:
#     fp.write((json.dumps(msg)).encode("utf-8"))

comp = stream_jsonl(COMPLETIONS)
last_task = None

# print(list(comp))

# for line in comp: 

#     if(not last_task):
#         last_task = line['task_iline']

#     while(line["task_id"] == last_task):

import json

scores = evaluate_functional_correctness(COMPLETIONS, k=[1, 3, 5], problem_file=PROBLEMS)

# Read the JSONL file and load each JSON object
data = []
with open(COMPLETIONS, 'r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Group all fields by task_id
grouped = {}
for entry in data:
    task_id = entry['task_id']
    # Create the group if it doesn't exist
    if task_id not in grouped:
        grouped[task_id] = []
    # Exclude the task_id from the individual completion entry to avoid redundancy
    completion_entry = {k: v for k, v in entry.items() if k != 'task_id'}
    grouped[task_id].append(completion_entry)

# Create a dictionary mapping task_id to pass@k value
score_dict = {list(score.keys())[0]: list(score.values())[0] for score in scores}

# Build the result in a more pythonic way using list comprehension
result = [
    {
        "task_id": task_id,
        "completions": completions,
        "pass@k": score_dict.get(task_id)
    }
    for task_id, completions in grouped.items()
]

# Print the result as a formatted JSON string
print(json.dumps(result, indent=2))



