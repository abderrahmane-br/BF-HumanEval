# from benchflow import BaseAgent
from openai import OpenAI
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, final
import os

import json
import uvicorn
from fastapi import FastAPI, HTTPException
from benchflow import BaseAgent

logger = logging.getLogger(__name__)

# class BaseAgent(ABC):
#     """
#     You need to extend this class to make your agent a server.
#     So that it can communicate with the benchmark client.
#     If you want to integrate your agent with BenchFlow, you need to implement the following methods:
#     ```
#     - call_api
#     ```
#     """
#     def __init__(self):
#         self.app = FastAPI()
#         self.setup_routes()

#     @final
#     def setup_routes(self):
#         """
#         Setup the routes for the agent.
#         """
#         @self.app.post("/response")
#         async def take_action(input_data: Dict[str, Any]):
#             try:
#                 if input_data.get("env_info") is not None:
#                     response = self.call_api(input_data.get("env_info"))
#                 elif input_data.get("input_data") is not None:
#                     response = self.call_api(input_data.get("input_data"))
#                 else:
#                     response = self.call_api(input_data)
#                 logger.info(f"[BaseAgent]: Got response from API: {response}")
#                 return response
#             except Exception as e:
#                 logger.error(f"[BaseAgent]: Error getting response: {str(e)}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.app.get("/")
#         async def root():
#             return {"message": "Welcome to Benchmarkthing Agent API"}

#     @final
#     def run_with_endpoint(self, host: str, port: int):
#         """
#         Run the agent server.
#         """
#         logger.info(f"Starting agent server on {host}:{port}")
#         uvicorn.run(self.app, host=host, port=port)

#     @abstractmethod 
#     def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
#         """
#         You can get the request information from the task_step_inputs parameter.
#         The task_step_inputs is a dictionary that contains the keys provided by the benchmark client.
#         You need to refer to the benchmark documentation to get the keys.

#         This method is called when the agent server receives a request from the benchmark client.
#         You need to implement this method to make your agent work and return the response to the benchmark client.
#         Your response could be a real action(e.g. click, scroll, etc) or just any prediction(e.g. code, text, etc) needed by the benchmark.
#         """
#         pass


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