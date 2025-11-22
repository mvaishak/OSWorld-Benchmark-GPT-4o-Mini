import base64
import json
from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np
import os
from datetime import datetime
import time

class GPT4oMiniAgent:
    def __init__(self, api_key, model="gpt-4o-mini", max_history=5):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_history = max_history
        self.action_history = []
        self.step_count = 0
        self.last_fail_reason = ""
        
        # Mapping common LLM key terms to pyautogui standard
        self.key_map = {
            "Return": "enter", "Enter": "enter", 
            "Control": "ctrl", "Ctrl": "ctrl",
            "Command": "command", "Cmd": "command",
            "Alt": "alt", "Shift": "shift",
            "Escape": "esc", "Tab": "tab",
            "Space": "space", "Up": "up", "Down": "down", "Left": "left", "Right": "right"
        }

    def _encode_image(self, image_data):
        # Robust image encoding handling various input types
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode("utf-8")
        else:
            # Fallback for unexpected types
            return "" 
            
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _normalize_key(self, key_name):
        # Clean up key names from LLM output
        clean_key = key_name.strip().replace("'", "").replace('"', "")
        return self.key_map.get(clean_key, clean_key.lower())

    def _build_system_prompt(self):
        return '''
You are an expert computer-use agent operating a Ubuntu desktop. 
Goal: Complete the user's high-level instruction.

INPUTS:
1. Screenshot: Visual state of the UI.
2. A11y Tree: Textual representation of UI elements (names, roles) to help identify icons/buttons.

RESPONSE FORMAT:
Respond ONLY with a valid JSON object.

ACTIONS:
1. CLICK: {"thought": "...", "action": {"type": "click", "x": 0.5, "y": 0.5}} 
   - Coordinates are [0,1] normalized (0,0 top-left).
2. TYPE: {"thought": "...", "action": {"type": "type", "text": "hello"}}
   - Use ONLY when a text field is already focused.
3. KEYPRESS: {"thought": "...", "action": {"type": "key", "keys": "enter"}}
   - For shortcuts use: "ctrl+c", "alt+f4".
4. SCROLL: {"thought": "...", "action": {"type": "scroll", "clicks": 2}}
   - Positive = Scroll UP, Negative = Scroll DOWN (PyAutoGUI standard is opposite to natural scrolling).
5. DONE: {"thought": "Task completed", "action": {"type": "done"}}
6. FAIL: {"thought": "Stuck", "action": {"type": "fail", "reason": "..."}}

STRATEGY:
- Check the A11y Tree to find the exact names of menus or icons if the screenshot is cluttered.
- If clicking a button, try to click the center.
- If a previous action failed, change strategy (e.g., use keyboard shortcuts instead of clicking).
'''

    def _build_user_prompt(self, instruction, history, accessibility_tree):
        # Provide a summarized version of the A11y tree if it's huge, or raw if manageable
        # For GPT-4o-mini context window, we might need to truncate if the tree is massive
        a11y_snippet = str(accessibility_tree)[:2000] + "..." if len(str(accessibility_tree)) > 2000 else str(accessibility_tree)

        prompt = f"User Instruction: {instruction}\n\n"
        
        if self.action_history:
            prompt += "Action History (Last 5):\n"
            for cmd in self.action_history[-5:]:
                prompt += f"- {cmd}\n"
        
        prompt += f"\nVisible UI Text (A11y Tree Snippet):\n{a11y_snippet}\n\n"
        
        prompt += "Based on the screenshot and UI text, output the next JSON action."
        return prompt

    def _convert_to_pyautogui(self, action_obj, w, h):
        atype = action_obj.get("type", "").lower()
        
        if atype == "click":
            x = min(1.0, max(0.0, float(action_obj.get("x", 0.5))))
            y = min(1.0, max(0.0, float(action_obj.get("y", 0.5))))
            px, py = int(x * (w - 1)), int(y * (h - 1))
            # Move then click ensures hover effects trigger which sometimes helps
            return f"pyautogui.moveTo({px}, {py}); pyautogui.sleep(0.2); pyautogui.click({px}, {py})"
            
        elif atype == "type":
            text = action_obj.get("text", "").replace("'", "\\'")
            return f"pyautogui.write('{text}', interval=0.05)"
            
        elif atype == "key":
            raw_keys = action_obj.get("keys", "")
            # Handle combos like "ctrl+s"
            if "+" in raw_keys:
                keys = [self._normalize_key(k) for k in raw_keys.split("+")]
                # Join them properly for hotkey: pyautogui.hotkey('ctrl', 's')
                args = "', '".join(keys)
                return f"pyautogui.hotkey('{args}')"
            else:
                k = self._normalize_key(raw_keys)
                return f"pyautogui.press('{k}')"

        elif atype == "scroll":
            clicks = int(action_obj.get("clicks", 0))
            # In many Linux VMs, scroll down is negative
            return f"pyautogui.scroll({clicks * -10})"
            
        elif atype == "done":
            return "DONE"
        elif atype == "fail":
            self.last_fail_reason = action_obj.get("reason", "Unknown")
            return "FAIL"
        
        return "FAIL"

    def act(self, observation, instruction):
        self.step_count += 1
        
        screenshot = observation.get("screenshot")
        # Extract text info (accessibility tree) required by the prompt
        a11y_tree = observation.get("accessibility_tree", "Not available")
        
        base64_img = self._encode_image(screenshot)
        if not base64_img:
            self.last_fail_reason = "Screenshot encoding failed"
            return "FAIL"

        # Determine resolution
        if isinstance(screenshot, np.ndarray):
            h, w = screenshot.shape[:2]
        elif isinstance(screenshot, Image.Image):
            w, h = screenshot.size
        else:
            w, h = 1920, 1080

        # API Call
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_user_prompt(instruction, self.action_history, a11y_tree)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }
        ]
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.1,
                    response_format={"type": "json_object"} # Force JSON mode
                )
                break
            except Exception as e:
                # Check if it's a rate limit error (usually contains '429')
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = base_delay * (2 ** attempt)  # 2s, 4s, 8s, 16s...
                    print(f"\n[Warning] Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # If it's another error (like Auth), crash immediately
                    print(f"Agent Error: {e}")
                    self.last_fail_reason = str(e)
                    return "FAIL"
        else:
            # This executes if the loop finishes without 'break' (max retries exceeded)
            print("Error: Max retries exceeded for Rate Limit.")
            self.last_fail_reason = "Rate limit exceeded (Max retries)"
            return "FAIL"
        try:
            content = response.choices[0].message.content
            print(f"\n[Agent Thought]: {content}") # Debugging
            
            data = json.loads(content)
            thought = data.get("thought")
            action = data.get("action")
            
            # Convert to PyAutoGUI string
            cmd_string = self._convert_to_pyautogui(action, w, h)
            
            # Loop detection (simple)
            if len(self.action_history) > 3 and all(x == cmd_string for x in self.action_history[-3:]):
                print("Loop detected. Failing.")
                self.last_fail_reason = "Loop detected"
                return "FAIL"
                
            self.action_history.append(cmd_string)
            return cmd_string

        except Exception as e:
            print(f"Agent Error: {e}")
            self.last_fail_reason = str(e)
            return "FAIL"

    def reset(self):
        self.action_history = []
        self.step_count = 0
        self.last_fail_reason = ""