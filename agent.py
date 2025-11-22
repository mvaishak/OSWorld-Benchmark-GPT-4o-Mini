import os
import json
import base64
import re
import logging
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# Configure logging
logger = logging.getLogger("GPT40MiniAgent")
logger.setLevel(logging.DEBUG)


class GPT40MiniAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.screen_width = 1920
        self.screen_height = 1080
        self.system_prompt = self._construct_system_prompt()

    def _construct_system_prompt(self) -> str:
        return f"""You are a precise computer control agent.
You interact with a computer running at {self.screen_width}x{self.screen_height} resolution (top-left is 0,0).
You are required to use `pyautogui` to perform the actions. These are the available actions:
AVAILABLE ACTION SPACE (Python syntax):
1. pyautogui.click(x, y): Left click at coordinates.
2. pyautogui.right_click(x, y): Right click at coordinates.
3. pyautogui.double_click(x, y): Double click at coordinates.
4. pyautogui.dragTo(x, y, duration=0.5): Drag to coordinates (note: use dragTo, not drag).
5. pyautogui.typewrite(text): Type string into active field.
6. pyautogui.press(key): Press key (e.g., 'enter', 'tab', 'esc').
7. pyautogui.hotkey(key1, key2): Key combo (e.g., 'ctrl', 'c').
8. WAIT: Pause execution for UI rendering.
9. FAIL: Signal task is impossible.
10. DONE: Signal successful task completion.

OUTPUT FORMAT:
You must output VALID JSON only. No conversational text. NO COMMENTS.
{{
  "thought": "Reasoning about the current state and next step...",
  "action": "pyautogui.click(500, 300)"
}}
"""

    def _clean_and_truncate_tree(self, tree_str: str, max_chars: int = 50000) -> str:
        """
        Heuristic to prevent Context Length Exceeded errors.
        1. Truncates very long trees (approx 12k tokens).
        2. Adds a warning if truncated.
        """
        if not tree_str:
            return "No accessibility info available."
        
        if len(tree_str) > max_chars:
            logger.warning(f"A11y Tree too large ({len(tree_str)} chars). Truncating to {max_chars}.")
            return tree_str[:max_chars] + "\n...[TRUNCATED DUE TO LENGTH]..."
        
        return tree_str

    def predict(self, instruction: str, observation: Dict[str, Any], history: List[Dict]) -> Tuple[Dict, str]:
        # 1. Visual Processing
        base64_image = self._encode_image(observation['screenshot'])

        # 2. Textual Processing (WITH NEW TRUNCATION)
        raw_tree = observation.get('accessibility_tree', '')
        # We limit to 50k chars (approx 12-15k tokens) to leave room for the image (~1k tokens) + history
        a11y_tree = self._clean_and_truncate_tree(raw_tree, max_chars=50000)

        # 3. Construct Multimodal Prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task: {instruction}\n"
                                f"Accessibility Tree:\n{a11y_tree}\n"
                                f"History: {json.dumps(history)}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.0
            )
            raw_content = response.choices[0].message.content
            logger.debug(f"Agent Trace: {raw_content}")
            
            return self._parse_response(raw_content)

        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            # Return a safe WAIT action if API fails so the loop doesn't crash
            return {"action": "WAIT"}, f"API_ERROR: {str(e)}"

    def _parse_response(self, content: str) -> Tuple[Dict, str]:
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")
            
            response_json = json.loads(match.group(0))
            
            action_str = response_json.get('action', '')
            # if not self._validate_action(action_str):
            #     logger.warning(f"Hallucination detected: Invalid action signature '{action_str}'")
            #     return {"action": "WAIT"}, content

            return response_json, content

        except Exception as e:
            logger.error(f"Parse Error: {str(e)}")
            return {"action": "WAIT"}, content


    def _encode_image(self, image_source) -> str:
        if isinstance(image_source, str) and os.path.exists(image_source):
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_source, bytes):
             return base64.b64encode(image_source).decode('utf-8')
        return ""