from typing import Dict, Any

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult

class HumanEvalBench(BaseBench):

    def __init__(self):
        super.__init__()

    def get_args(self, task_id: str) -> BenchArgs:
        pass
    
    def get_image_name(self) -> str:
        pass

    def get_results_dir_in_container(self) -> str:
        pass

    def get_log_files_dir_in_container(self) -> str:
        pass

    def get_result(self, task_id: str) -> BenchmarkResult:
        pass

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        pass