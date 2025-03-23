import os
import json
from typing import Dict, Any

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult

class HumanEvalBench(BaseBench):

    def __init__(self):
        super.__init__()

    def get_args(self, task_id: str) -> BenchArgs:
        arguments = {
            "required": ["K_VALUES", "N_COMPLETIONS"],
            "optional": ["BASE_URL", "MODEL"]
        }
        return BenchArgs(arguments)
    
    def get_image_name(self) -> str:
        # return "abderrahmane-br/benchflow:humaneval-v1"
        pass

    def get_results_dir_in_container(self) -> str:
        return "/app/eval_results"

    def get_log_files_dir_in_container(self) -> str:
        return "/app/logs"

    def get_result(self, task_id: str) -> BenchmarkResult:

        result_file = os.path.join(self.results_dir, "humaneval_result.json")
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)

            log = ''.join(json.dumps(item, ensure_ascii=False) for item in result)

            def compute_average_pass_k(result):
                """
                calculates the average pass@k scores
                across all tasks. Returns a dictionary with each pass@k key and its average value.
                """
                
                # Dictionaries to sum values and count occurrences for each pass@k key
                sums = {}
                counts = {}
                
                for task in result:
                    pass_k = task.get("pass@k", {})
                    for key, value in pass_k.items():
                        sums[key] = sums.get(key, 0) + value
                        counts[key] = counts.get(key, 0) + 1
                
                # Compute average for each key
                averages = {key: sums[key] / counts[key] for key in sums}
                return averages
            
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=True,
                metrics={"score": compute_average_pass_k(result)},
                log={"result": log},
                other={},
            )
        except Exception as e:
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                metrics={"score": 0},
                log={"error": str(e)},
                other={"error": str(e)},
            )

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        # Only one task for HumanEval benchmark
        return {"task_ids": ["0"], "error_message": None}