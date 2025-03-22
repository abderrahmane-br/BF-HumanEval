# from benchflow import BaseAgent
from openai import OpenAI
import logging
import os

from benchflow import BaseAgent

logger = logging.getLogger(__name__)


class HumanEvalAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.system_instruction = (
            """You are a Python code completion agent.
            Your task is to complete a given incomplete Python code snippet so that the final program is correct, coherent, and fully functional.
            
            RULES: 
            1. Produce only valid Python code.
            2. Continue exactly from where the provided code ends. Do not repeat or modify the code that has already been given.
            3. Do not include any text, explanations, or comments outside the Python code.
            4. Ensure your completion follows Python best practices and integrates seamlessly with the given code.
            5. Complete the code as effectively as possible.
            """
        )

    def get_completions(self, 
                        problems, 
                        base_url=None,
                        model="gpt-4o", 
                        n_completions=1):

        client = OpenAI(base_url=base_url, 
                        api_key=self.api_key)
        
        messages = [
            {
                "role": "system",
                "content": self.system_instruction
            }
        ]
        completions = []

        for prb in problems:
            task_id = prb["task_id"]
            prompt = prb["prompt"]

            messages.append({
                "role": "user",
                "content": prompt
            })

            # generate one or more completions
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                n=n_completions
            )

            for choice in completion :
                completions.append({
                    "task_id": task_id,
                    "completion": choice.message.content
                })

        return completions
    

    def call_api(self, task_step_inputs):

        problems = task_step_inputs["problems"]
        n = task_step_inputs["n_completions"]
        base_url = task_step_inputs["api"]["base_url"]
        model = task_step_inputs["api"]["model"]

        try:
            logger.info(f"[HumanEvalAgent]: Calling OpenAI API")
            completions = self.get_completions(problems, base_url, model, n)
            logger.info(f"[HumanEvalAgent]: Generated completions: {completions}")
            return completions
        
        except Exception as e:
            logger.error(f"[HumanEvalAgent]: Error calling OpenAI API: {e}")
            raise


if __name__ == "__main__":
    agent = HumanEvalAgent()
    agent.run_with_endpoint("0.0.0.0", 10004)