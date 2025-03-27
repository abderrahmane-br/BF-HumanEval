from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import argparse
from os import path, makedirs
import numpy as np
import tqdm
import json

from utils.utils import read_problems, estimate_pass_at_k, post_log
from utils.execution import check_correctness

from benchflow import BenchClient

# problems path will be provided as CLI arg
# results folder will be provided as CLI arg


def evaluate_functional_correctness(
    problems,
    completions,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples"
    """
    payload = {
            "data": {
                "message": "[HUMANEVAL Client]: evaluating functional correctness" 
            }
        }
    post_log(payload)


    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(json.loads(completions)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    scores = []
    for result in results.values():
        total, correct = [], [],
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
        total_np = np.array(total)
        correct_np = np.array(correct)

        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total_np, correct_np, k).mean().item()
                    for k in ks if (total_np >= k).all()}
        scores.append({result[0][1]["task_id"]: pass_at_k})
    print(scores)
    # Finally, save the results in one file:
    def combine_results():
        for sample in json.loads(completions):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    return combine_results(), scores


def evaluate(intelligence_url, problem_file, k, n_completions, out_dir):

    payload = {
            "data": {
                "message": "[HUMANEVAL Client]: evaluating" 
            }
        }
    post_log(payload)

    bench_client = HumanEvalClient(intelligence_url)
    problems = read_problems(problem_file)

    env = {
        "problems": problems,
        "k": k,
        "n_completions": n_completions
    }

    # get completions from the agent
    response = bench_client.get_response(env)
    completions = response["raw_response"]

    # evaluate the completions and get results and pass@k scores
    results, scores = evaluate_functional_correctness(problems, completions, k)

    formatted_result = format_result(results, scores)
    # print("formatted res ", formatted_result)
    save_result(formatted_result, out_dir)

def format_result(results, scores):

    payload = {
            "data": {
                "message": "[HUMANEVAL Client]: formatting result" 
            }
        }
    post_log(payload)

    # Group all fields by task_id
    grouped = {}
    for entry in results:
        task_id = entry['task_id']
        # Create the group if it doesn't exist
        if task_id not in grouped:
            grouped[task_id] = []
        # Exclude the task_id from the individual completion entry to avoid redundancy
        completion_entry = {k: v for k, v in entry.items() if k != 'task_id'}
        grouped[task_id].append(completion_entry)

    # Create a dictionary mapping task_id to pass@k value
    score_dict = {list(score.keys())[0]: list(score.values())[0] for score in scores}

    # Build the final result
    result = [
        {
            "task_id": task_id,
            "completions": completions,
            "pass@k": score_dict.get(task_id)
        }
        for task_id, completions in grouped.items()
    ]

    return result




def save_result(result, out_dir):
    payload = {
            "data": {
                "message": "[HUMANEVAL Client]: saving result" 
            }
        }
    post_log(payload)
    # print('result ', result)
    output_file_path = path.join(out_dir, "humaneval_results.json")
    with open(output_file_path, "wb") as fp:
        fp.write(json.dumps(result).encode('utf-8'))

class HumanEvalClient(BenchClient):

    def __init__(self, intelligent_url):
        super().__init__(intelligent_url, 3)

    def prepare_input(self, raw_input_data):
        return raw_input_data
    
    def parse_response(self, raw_response):
        # print("this is the resp ", (raw_response))
        return {"response" : raw_response}
    

if __name__ == "__main__":
    # evaluate(
    #     intelligence_url="http://0.0.0.0:10004",
    #     problem_file="/workspaces/benchflow/human-eval/data/example_problem.jsonl",
    #     k=[1, 3],
    #     n_completions=6,
    #     out_file_path="./output.json"
    # )
    payload = {
            "data": {
                "message": "[HUMANEVAL Client]: running" 
            }
        }
    post_log(payload)


    parser = argparse.ArgumentParser()

    parser.add_argument("--intelligence_url", type=str)
    parser.add_argument("--problems_file", type=str)
    parser.add_argument("--k", nargs="+", type=int)
    parser.add_argument("--n_completions", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--base_url", type=str, required=False)
    parser.add_argument("--model", type=str, required=False)

    args = parser.parse_args()

    makedirs(args.output_dir, exist_ok=True)

    evaluate(args.intelligence_url, args.problems_file, args.k, args.n_completions, args.output_dir)