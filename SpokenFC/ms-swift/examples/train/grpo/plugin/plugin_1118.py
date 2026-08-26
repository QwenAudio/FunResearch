import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# For additional reward functions, refer to swift/plugin/orm.py.
class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM


class MultiTurnThinkingTips(ORM):
    """
    A reward function example designed for use with the `ThinkingTipsScheduler`.

    This class demonstrates how to handle reward computation when a single
    training sample (or request) is split into multiple "turns" or steps.
    Specifically, it computes the reward based on the **last turn** of each
    multi-turn trajectory using a math accuracy function.

    NOTE
    ----
    If you feed fragments of the *same* trajectory as independent samples, this
    function **must return an identical reward for every fragment**
    """

    def __init__(self):
        from swift.plugin.orm import MathAccuracy
        self.acc_func = MathAccuracy()

    def __call__(self, completions, **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]
            # For reward calculation, we use the entire trajectory of this sample.
            # Here, we specifically evaluate only the last turn.
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']
            last_turn_solution = total_trajectory_inputs[-1]['solution']
            # Compute reward based on math accuracy for the final completion.
            reward = self.acc_func([last_turn_completion], [last_turn_solution])[0]
            rewards.append(reward)
        return rewards


orms['thinking_tips'] = MultiTurnThinkingTips


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager



import re
import ast
from typing import Tuple, Dict, Any, List

# 假设 orms, ORM, 和 logger 已经定义好
# from swift.llm import orms, ORM
# from swift.utils import get_logger
# logger = get_logger()


class SpokenFunctionCallReward_V1(ORM):
    """
    一个用于评估模型生成“函数调用”字符串质量的自定义奖励函数。
    """

    @staticmethod
    def _parse_function_call(call_string: str) -> Tuple[str | None, Dict[str, Any]]:
        """
        辅助函数：安全地解析一个类似函数调用的字符串。
        (此函数代码未变)
        """
        call_string = call_string.strip()
        match = re.match(r'(\w+)\((.*)\)', call_string, re.DOTALL)
        if not match:
            return None, {}

        func_name, args_str = match.groups()
        if not args_str.strip():
            return func_name, {}

        try:
            tree = ast.parse(f"dummy_func({args_str})", mode='eval')
            call_node = tree.body

            if isinstance(call_node, ast.Call):
                kwargs = {}
                for keyword in call_node.keywords:
                    try:
                        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                    except ValueError:
                        pass
                return func_name, kwargs

            return func_name, {}
        except Exception:
            return func_name, {}

    # ================================================================= #
    # =====================  核 心 修 改 在 这 里  ===================== #
    # ================================================================= #
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Swift框架会调用此方法来计算奖励。

        Args:
            completions (List[str]): 模型生成的一批输出。
            **kwargs: 数据集中的其他列，批处理后的形式。
                      我们将从 kwargs['solution'] 中直接获取标准答案。

        Returns:
            List[float]: 包含每个样本奖励分数的列表。
        """
        rewards = []
        
        # 直接从 kwargs 中获取批处理后的 solution 数据
        batched_solutions = kwargs.get('solution')
        if not batched_solutions or len(completions) != len(batched_solutions):
            logger.error("Data mismatch: 'completions' and 'solution' have different lengths.")
            return [-1.0] * len(completions)

        # 遍历批次中的每一个样本
        for i, response in enumerate(completions):
            # 直接通过索引获取当前样本的标准答案 (ground_truth)
            ground_truth = batched_solutions[i]

            # 如果在数据中找不到标准答案，给予最低奖励
            if not ground_truth:
                rewards.append(-1.0)
                continue
            
            # 使用 logger 记录模型输出和标准答案，便于调试
            logger.info(f"[Sample {i}] Model Response: '{response}'")
            logger.info(f"[Sample {i}] Ground Truth:   '{ground_truth}'")

            # 步骤 1: 解析模型输出 (response) 和标准答案 (ground_truth)
            gen_name, gen_params = self._parse_function_call(response)
            gt_name, gt_params = self._parse_function_call(ground_truth)

            if not gen_name or not gt_name:
                rewards.append(-1.0)
                continue

            # 步骤 2: 计算各部分得分
            r_name = 1.0 if gen_name == gt_name else 0.0

            gen_keys = set(gen_params.keys())
            gt_keys = set(gt_params.keys())
            if not gt_keys:
                r_keys = 1.0 if not gen_keys else 0.0
            else:
                intersection_keys = len(gen_keys.intersection(gt_keys))
                union_keys = len(gen_keys.union(gt_keys))
                r_keys = intersection_keys / union_keys if union_keys > 0 else 1.0

            common_keys = gen_keys.intersection(gt_keys)
            if not common_keys:
                r_values = 1.0 if not gt_keys else 0.0
            else:
                matching_values = 0
                for key in common_keys:
                    if gen_params.get(key) == gt_params.get(key):
                        matching_values += 1
                r_values = matching_values / len(common_keys)

            # 步骤 3: 加权计算最终奖励
            final_reward = (r_name + r_keys + r_values - 1.5) * 2.0 / 3.0
            rewards.append(final_reward)

        return rewards


# 向Swift框架的ORM注册表中注册我们自定义的奖励函数
orms['sfc_reward'] = SpokenFunctionCallReward_V1



class SpokenFunctionCallReward_V2(ORM):
    """
    一个用于评估模型生成“函数调用”字符串列表质量的自定义奖励函数。
    
    此版本支持将字符串化的列表（例如 "[\"func1()\", \"func2()\"]"）
    与标准答案的类似列表进行比较，并计算一个综合的 F1-like 奖励。
    """

    @staticmethod
    def _parse_function_call(call_string: str) -> Tuple[str | None, Dict[str, Any]]:
        """
        辅助函数：安全地解析一个类似函数调用的字符串。
        (此函数代码未变)
        """
        call_string = call_string.strip()
        match = re.match(r'(\w+)\((.*)\)', call_string, re.DOTALL)
        if not match:
            return None, {}

        func_name, args_str = match.groups()
        if not args_str.strip():
            return func_name, {}

        try:
            # 使用 f-string 构造一个可解析的表达式
            tree = ast.parse(f"dummy_func({args_str})", mode='eval')
            call_node = tree.body

            if isinstance(call_node, ast.Call):
                kwargs = {}
                # 只解析关键字参数 (kwargs)
                for keyword in call_node.keywords:
                    try:
                        # ast.literal_eval 只能安全地评估字面量
                        # (如果值是 AST 节点，ast.literal_eval 可以处理)
                        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                    except ValueError:
                        # 如果值不是字面量（例如，一个变量名），则跳过
                        pass
                return func_name, kwargs

            # 如果解析结果不是一个调用
            return func_name, {}
        except Exception as e:
            # 捕获所有解析异常（例如语法错误，包括 'leading zeros'）
            # 这可能发生在 args_str 格式错误时 (例如 key=09:00)
            logger.warning(f"Failed to parse args_str: '{args_str}'. Error: {e}")
            return func_name, {}

    # ================================================================= #
    # =====================    核 心 修 改 在 这 里   ===================== #
    # ================================================================= #
    def _parse_call_list(self, call_list_string: str) -> List[str]:
        """
        辅助函数：安全地将字符串表示的列表解析为字符串列表。
        
        策略：
        1. 尝试使用 `ast.literal_eval` 进行严格解析（适用于格式正确的字符串）。
        2. 如果失败（例如由于内部引号未转义导致 SyntaxError），
           则回退到使用正则表达式查找所有 `func(...)` 模式。
        """
        if not call_list_string or call_list_string.strip() == '[]':
            return []
        
        call_list_string = call_list_string.strip()

        # 策略 1: 尝试严格解析 (适用于模型生成的带转义的有效字符串)
        try:
            parsed_list = ast.literal_eval(call_list_string)
            
            if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                return parsed_list
            elif isinstance(parsed_list, str):
                 # 处理模型可能只返回一个字符串而不是列表的情况
                 # 检查它是否像一个函数调用
                 if re.match(r'\w+\(.*\)', parsed_list, re.DOTALL):
                     return [parsed_list]
                 
        except (ValueError, SyntaxError, TypeError) as e:
            # 策略 2: 回退到正则表达式 (适用于数据集中格式错误的字符串)
            # 这个 SyntaxError 就是你日志中看到的 "leading zeros" 错误
            logger.warning(f"ast.literal_eval failed: '{e}'. Falling back to regex parser for string: '{call_list_string}'")
            
            try:
                # 这个正则表达式查找所有 `func_name(...)` 模式，
                # 并处理一级嵌套的括号 `()`，但忽略 `[]` 或 `{}`。
                # 这对于解析格式错误的字符串通常足够健壮。
                # \w+\(           - func(
                # (?:             - non-capturing group for content
                #     [^()]        - any char that is not ( or )
                #     |            - OR
                #     \([^)]*\)    - a nested group ()
                # )* - zero or more times
                # \)              - )
                calls = re.findall(r'(\w+\((?:[^()]|\([^)]*\))*\))', call_list_string)
                if calls:
                    return calls
                else:
                    # 如果 regex 也失败了，但字符串看起来像一个函数
                    # (例如，括号不平衡导致 regex 失败)
                    # 我们做一个最后的猜测
                    if re.match(r'\w+\(.*\)', call_list_string, re.DOTALL):
                         return [call_list_string]
                         
            except Exception as regex_e:
                logger.error(f"Regex fallback parser also failed: {regex_e} on string: '{call_list_string}'")
                return []

        # 如果 ast.literal_eval 成功了但返回了非列表/字符串，或 regex 失败了
        logger.warning(f"Could not parse call list string: '{call_list_string}'")
        return []
    # ================================================================= #
    # ================================================================= #

    def _calculate_single_call_reward(self, gen_call: str, gt_call: str) -> float:
        """
        计算 *单个* 生成的函数调用与 *单个* 标准答案函数调用之间的奖励。
        奖励范围在 [-1.0, 1.0] 之间。
        (此函数代码未变)
        """
        # 步骤 1: 解析模型输出 (gen_call) 和标准答案 (gt_call)
        gen_name, gen_params = self._parse_function_call(gen_call)
        gt_name, gt_params = self._parse_function_call(gt_call)

        # 步骤 1.5: 处理解析失败
        
        # 如果标准答案 (GT) 无法解析（例如数据本身是空的或无效的）
        if not gt_name:
            # 如果模型也生成了无效/不可解析的调用，则视为“正确”
            # 如果模型生成了有效的调用，而 GT 是无效的，则视为“错误”
            return 1.0 if not gen_name else -1.0
        
        # 如果 GT 是有效的，但模型生成的是无效的
        if not gen_name:
            return -1.0

        # 步骤 2: 计算各部分得分 (此时 gen_name 和 gt_name 都有效)
        
        # 2.1: 函数名称得分
        r_name = 1.0 if gen_name == gt_name else 0.0

        # 2.2: 参数键 (Keys) 得分 (Jaccard 相似度)
        gen_keys = set(gen_params.keys())
        gt_keys = set(gt_params.keys())
        
        if not gt_keys:
            # 如果 GT 不需要参数，那么模型也不应该提供参数
            r_keys = 1.0 if not gen_keys else 0.0
        else:
            intersection_keys = len(gen_keys.intersection(gt_keys))
            union_keys = len(gen_keys.union(gt_keys))
            # Jaccard 相似度
            r_keys = intersection_keys / union_keys if union_keys > 0 else 1.0 # (如果 union 为0，说明两者都为空，Jaccard 为 1)

        # 2.3: 参数值 (Values) 得分 (基于共同的键)
        common_keys = gen_keys.intersection(gt_keys)
        
        if not common_keys:
            # 如果没有共同的参数键
            # 如果 GT 本来就不需要参数，那么值匹配得分为 1.0
            # 如果 GT 需要参数，但模型一个都没匹配上，得分为 0.0
            r_values = 1.0 if not gt_keys else 0.0
        else:
            matching_values = 0
            for key in common_keys:
                # 比较共同键的值是否相等
                if gen_params.get(key) == gt_params.get(key):
                    matching_values += 1
            r_values = matching_values / len(common_keys)

        # 步骤 3: 加权计算最终奖励 (归一化到 [-1.0, 1.0])
        total_score = r_name + r_keys + r_values
        final_reward = (total_score - 1.5) * 2.0 / 3.0
        
        return final_reward


    def __call__(self, completions: List[str], **kwargs) -> List[float]:
            """
            Swift框架会调用此方法来计算奖励。
            (此函数代码已修改以支持多轮数据结构)
            """
            rewards = []
            
            # 直接从 kwargs 中获取批处理后的 solution 数据
            # 在多轮场景下, 'solution' 键通常包含一个 dict 列表 (来自 JSON)
            batched_solutions = kwargs.get('solution')
            if not batched_solutions or len(completions) != len(batched_solutions):
                logger.error(f"Data mismatch: 'completions' (len {len(completions)}) and 'solution' (len {len(batched_solutions)}) have different lengths.")
                return [-1.0] * len(completions)

            # 定义一个基础奖励
            EMPTY_MATCH_REWARD = -1.0 # (r_name=0, r_keys=0, r_values=0) -> (0-1.5)*2/3 = -1.0

            # 遍历批次中的每一个样本
            for i, response_str in enumerate(completions):
                # response_str (completions[i]) 已经是模型生成的【最后一轮响应】
                
                # =====================    核 心 修 改 在 这 里   ===================== #
                
                # 获取当前样本的标准答案 (ground_truth) 数据包
                sample_data = batched_solutions[i]
                gt_str = None

                # 从数据包(dict)中提取 'solution' 字段
                if isinstance(sample_data, dict):
                    gt_str = sample_data.get('solution')
                elif isinstance(sample_data, str):
                    # 保留后备逻辑, 以防 'solution' 已经是字符串列表
                    gt_str = sample_data
                
                # ================================================================= #

                # 如果在数据中找不到标准答案 (gt_str 为 None 或空字符串)
                if not gt_str:
                    logger.warning(f"[Sample {i}] Could not extract 'solution' string from batched data. Assigning -1.0.")
                    rewards.append(-1.0)
                    continue
                
                # 使用 logger 记录原始输入，便于调试
                # response_str 是模型的最后一轮输出
                logger.info(f"[Sample {i}] Model Response: '{response_str}'")
                # gt_str 是 'solution' 字段的内容
                logger.info(f"[Sample {i}] Ground Truth:   '{gt_str}'")

                # 步骤 1: 将字符串解析为函数调用列表 (使用我们新的鲁棒解析器)
                gen_list = self._parse_call_list(response_str)
                gt_list = self._parse_call_list(gt_str)

                logger.info(f"[Sample {i}] Parsed Gen List: {gen_list}")
                logger.info(f"[Sample {i}] Parsed GT List:   {gt_list}")

                # 步骤 2: 计算 F1-like 奖励 (Precision 和 Recall 的平均值)

                # 2.1: 计算 Precision
                if not gen_list:
                    avg_precision = 1.0
                else:
                    precision_rewards = []
                    for g_call in gen_list:
                        best_r_for_g = max(
                            [self._calculate_single_call_reward(g_call, t_call) for t_call in gt_list] 
                            or [EMPTY_MATCH_REWARD] # 如果 gt_list 为空
                        )
                        precision_rewards.append(best_r_for_g)
                    
                    avg_precision = sum(precision_rewards) / len(gen_list)

                # 2.2: 计算 Recall
                if not gt_list:
                    avg_recall = 1.0
                else:
                    recall_rewards = []
                    for t_call in gt_list:
                        best_r_for_t = max(
                            [self._calculate_single_call_reward(g_call, t_call) for g_call in gen_list] 
                            or [EMPTY_MATCH_REWARD] # 如果 gen_list 为空
                        )
                        recall_rewards.append(best_r_for_t)
                    
                    avg_recall = sum(recall_rewards) / len(gt_list)

                # 步骤 3: 计算最终奖励
                final_reward = (avg_precision + avg_recall) / 2.0
                
                logger.info(f"[Sample {i}] AvgPrecision: {avg_precision:.4f}, AvgRecall: {avg_recall:.4f}, Final Reward: {final_reward:.4f}")
                rewards.append(final_reward)

            return rewards


# 向Swift框架的ORM注册表中注册我们自定义的奖励函数
orms['sfc_reward_list'] = SpokenFunctionCallReward_V2




import re
import ast
import logging
from typing import List, Tuple, Dict, Any

# 假设 ORM, orms, logger 已经按Swift框架的要求在别处定义
# (例如: logger = logging.getLogger(__name__))
# class ORM: pass
# orms = {}

class SpokenFunctionCallReward_V3(ORM):
    """
    一个用于评估模型生成“函数调用”字符串列表质量的自定义奖励函数。
    
    [修改版]：此版本现在会首先尝试从 <answer>...</answer> 标签中提取内容，
    然后再对提取的内容执行函数调用解析和 F1 奖励计算。
    """

    @staticmethod
    def _parse_function_call(call_string: str) -> Tuple[str | None, Dict[str, Any]]:
        """
        辅助函数：安全地解析一个类似函数调用的字符串。
        (此函数代码未变)
        """
        call_string = call_string.strip()
        match = re.match(r'(\w+)\((.*)\)', call_string, re.DOTALL)
        if not match:
            return None, {}

        func_name, args_str = match.groups()
        if not args_str.strip():
            return func_name, {}

        try:
            # 使用 f-string 构造一个可解析的表达式
            tree = ast.parse(f"dummy_func({args_str})", mode='eval')
            call_node = tree.body

            if isinstance(call_node, ast.Call):
                kwargs = {}
                # 只解析关键字参数 (kwargs)
                for keyword in call_node.keywords:
                    try:
                        # ast.literal_eval 只能安全地评估字面量
                        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                    except ValueError:
                        # 如果值不是字面量（例如，一个变量名），则跳过
                        pass
                return func_name, kwargs

            # 如果解析结果不是一个调用
            return func_name, {}
        except Exception as e:
            # 捕获所有解析异常
            logger.warning(f"Failed to parse args_str: '{args_str}'. Error: {e}")
            return func_name, {}

    def _parse_call_list(self, call_list_string: str) -> List[str]:
        """
        辅助函数：安全地将字符串表示的列表解析为字符串列表。
        (此函数代码未变)
        """
        if not call_list_string or call_list_string.strip() == '[]':
            return []
        
        call_list_string = call_list_string.strip()

        # 策略 1: 尝试严格解析
        try:
            parsed_list = ast.literal_eval(call_list_string)
            
            if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                return parsed_list
            elif isinstance(parsed_list, str):
                 if re.match(r'\w+\(.*\)', parsed_list, re.DOTALL):
                     return [parsed_list]
                 
        except (ValueError, SyntaxError, TypeError) as e:
            # 策略 2: 回退到正则表达式
            logger.warning(f"ast.literal_eval failed: '{e}'. Falling back to regex parser for string: '{call_list_string}'")
            
            try:
                calls = re.findall(r'(\w+\((?:[^()]|\([^)]*\))*\))', call_list_string)
                if calls:
                    return calls
                else:
                    if re.match(r'\w+\(.*\)', call_list_string, re.DOTALL):
                         return [call_list_string]
                         
            except Exception as regex_e:
                logger.error(f"Regex fallback parser also failed: {regex_e} on string: '{call_list_string}'")
                return []

        logger.warning(f"Could not parse call list string: '{call_list_string}'")
        return []

    def _calculate_single_call_reward(self, gen_call: str, gt_call: str) -> float:
        """
        计算 *单个* 生成的函数调用与 *单个* 标准答案函数调用之间的奖励。
        (此函数代码未变)
        """
        # 步骤 1: 解析
        gen_name, gen_params = self._parse_function_call(gen_call)
        gt_name, gt_params = self._parse_function_call(gt_call)

        # 步骤 1.5: 处理解析失败
        if not gt_name:
            return 1.0 if not gen_name else -1.0
        
        if not gen_name:
            return -1.0

        # 步骤 2: 计算各部分得分
        r_name = 1.0 if gen_name == gt_name else 0.0

        gen_keys = set(gen_params.keys())
        gt_keys = set(gt_params.keys())
        
        if not gt_keys:
            r_keys = 1.0 if not gen_keys else 0.0
        else:
            intersection_keys = len(gen_keys.intersection(gt_keys))
            union_keys = len(gen_keys.union(gt_keys))
            r_keys = intersection_keys / union_keys if union_keys > 0 else 1.0

        common_keys = gen_keys.intersection(gt_keys)
        
        if not common_keys:
            r_values = 1.0 if not gt_keys else 0.0
        else:
            matching_values = 0
            for key in common_keys:
                if gen_params.get(key) == gt_params.get(key):
                    matching_values += 1
            r_values = matching_values / len(common_keys)

        # 步骤 3: 加权计算
        total_score = r_name + r_keys + r_values
        final_reward = (total_score - 1.5) * 2.0 / 3.0
        
        return final_reward

    # ================================================================= #
    # =====================    __call__ 已被修改   ===================== #
    # ================================================================= #
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Swift框架会调用此方法来计算奖励。
        [修改版]：现在会先从 <answer> 标签提取内容。
        """
        rewards = []
        
        # 直接从 kwargs 中获取批处理后的 solution 数据
        batched_solutions = kwargs.get('solution')
        if not batched_solutions or len(completions) != len(batched_solutions):
            logger.error(f"Data mismatch: 'completions' (len {len(completions)}) and 'solution' (len {len(batched_solutions)}) have different lengths.")
            return [-1.0] * len(completions)

        # 定义一个基础奖励
        EMPTY_MATCH_REWARD = -1.0 # (r_name=0, r_keys=0, r_values=0) -> (0-1.5)*2/3 = -1.0

        # 遍历批次中的每一个样本
        for i, response_str_raw in enumerate(completions):
            # 获取当前样本的 *原始* 标准答案 (ground_truth) 字符串
            gt_str_raw = batched_solutions[i]

            # 如果在数据中找不到标准答案，给予最低奖励
            if not gt_str_raw:
                rewards.append(-1.0)
                continue
            
            # ========================================================= #
            # 步骤 0: 从 <answer> 标签中提取内容
            # ========================================================= #
            
            # 1. 提取标准答案 (使用 re.DOTALL 以匹配多行)
            sol_match = re.search(r'<answer>(.*?)</answer>', gt_str_raw, re.DOTALL)
            gt_str = sol_match.group(1).strip() if sol_match else gt_str_raw.strip()
            
            # 2. 提取模型输出 (使用 re.DOTALL 以匹配多行)
            content_match = re.search(r'<answer>(.*?)</answer>', response_str_raw, re.DOTALL)
            response_str = content_match.group(1).strip() if content_match else response_str_raw.strip()
            
            # ========================================================= #

            # 使用 logger 记录 *提取后* 的输入，便于调试
            logger.info(f"[Sample {i}] Extracted Model Response: '{response_str}'")
            logger.info(f"[Sample {i}] Extracted Ground Truth:   '{gt_str}'")

            # 步骤 1: 将 *提取后* 的字符串解析为函数调用列表 (使用我们新的鲁棒解析器)
            gen_list = self._parse_call_list(response_str)
            gt_list = self._parse_call_list(gt_str)

            logger.info(f"[Sample {i}] Parsed Gen List: {gen_list}")
            logger.info(f"[Sample {i}] Parsed GT List:   {gt_list}")

            # 步骤 2: 计算 F1-like 奖励 (Precision 和 Recall 的平均值)

            # 2.1: 计算 Precision
            if not gen_list:
                # 如果模型未生成任何调用
                # 如果 GT 也不需要调用，则P=1.0；如果 GT 需要调用，则P=1.0 (因为没有“错误”的调用)
                # 按照原始逻辑，此处设为 1.0 是合理的
                avg_precision = 1.0
            else:
                precision_rewards = []
                for g_call in gen_list:
                    best_r_for_g = max(
                        [self._calculate_single_call_reward(g_call, t_call) for t_call in gt_list] 
                        or [EMPTY_MATCH_REWARD] # 如果 gt_list 为空
                    )
                    precision_rewards.append(best_r_for_g)
                
                avg_precision = sum(precision_rewards) / len(gen_list)

            # 2.2: 计算 Recall
            if not gt_list:
                # 如果 GT 不需要任何调用
                # 如果模型也没生成，则R=1.0；如果模型生成了，则R=1.0 (因为没有“需要”的调用)
                avg_recall = 1.0
            else:
                recall_rewards = []
                for t_call in gt_list:
                    best_r_for_t = max(
                        [self._calculate_single_call_reward(g_call, t_call) for g_call in gen_list] 
                        or [EMPTY_MATCH_REWARD] # 如果 gen_list 为空
                    )
                    recall_rewards.append(best_r_for_t)
                
                avg_recall = sum(recall_rewards) / len(gt_list)

            # 步骤 3: 计算最终奖励
            final_reward = (avg_precision + avg_recall) / 2.0
            
            logger.info(f"[Sample {i}] AvgPrecision: {avg_precision:.4f}, AvgRecall: {avg_recall:.4f}, Final Reward: {final_reward:.4f}")
            rewards.append(final_reward)

        return rewards

# 向Swift框架的ORM注册表中注册我们自定义的奖励函数
# 建议使用一个新名字注册，以区分原始版本
orms['sfc_reward_list_cot'] = SpokenFunctionCallReward_V3