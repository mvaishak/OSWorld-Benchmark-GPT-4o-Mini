import base64
import json
from io import BytesIO

from openai import OpenAI
from PIL import Image
import numpy as np
import pyautogui

import os
import subprocess
from datetime import datetime


class GPT4oMiniAgent:
    """
    Improved GPT-4o-mini computer-use agent for OSWorld.

    Changes vs. original:
    - Model returns structured actions (type + args), we convert to OSWorld strings.
    - Uses normalized coordinates [0,1] in prompt, mapped to 0–1000 for OSWorld.
    - Lower temperature for stability.
    - Basic loop detection to avoid repeating the same action forever.
    """

    def __init__(self, api_key, model="gpt-4o-mini", max_history=5):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_history = max_history
        self.action_history = []
        self.step_count = 0
    def _save_debug_screenshot(self, image_data, step):
        """
        Saves the current screenshot to disk and opens it on Mac.
        """
        # Ensure directory exists
        save_dir = "debug_screenshots"
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{save_dir}/step_{step}_{timestamp}.png"

        # Convert bytes to PIL Image if necessary
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data))
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data # Assume it's already a PIL Image

        # Save
        #image.save(filename)
        print(f"[DEBUG] Saved screenshot to: {filename}")

        # MAC SPECIFIC: Open immediately in Preview
        # (Comment this out if you don't want windows popping up constantly)
        try:
            subprocess.run(["open", filename]) 
        except Exception as e:
            print(f"Could not open image: {e}")
            
        return image

    def _encode_image(self, image_data):
        """Convert numpy array / PIL image / bytes to base64 PNG."""
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_system_prompt(self):
        """Build the system prompt that explains the agent's role and capabilities."""

        return '''  
        You are an expert computer-use agent operating a Linux desktop in the OSWorld environment.  
        Your goal is to complete the user’s instruction by interacting with the desktop through precise actions: mouse clicks, keyboard input, typing, and scrolling.

        At each step:
        - You receive a screenshot observation (image) and the high-level user instruction.

        Rules:
        - Analyze the current screenshot and progress toward the task.
        - Only select one action at a time, based on what is currently visible on the screen.
        - Never repeat failed actions; always adjust based on what changed (or did not) after your previous step.
        - Before acting, reason step-by-step: briefly state your assessment and which UI element or strategy you are targeting.

        Available Actions (ALWAYS respond in this JSON schema—no extra text, no markdown):

        {
        "thought": "Brief reasoning about the current UI state, what changed since last step, and next move.",
        "action": {
            "type": "click", "x": <float between 0 and 1>, "y": <float between 0 and 1>
        }
        }

        or

        {
        "thought": "Text entry needed; the input box is now focused.",
        "action": { "type": "type", "text": "..." }
        }

        or

        {
        "thought": "A keyboard shortcut is required.",
        "action": { "type": "key", "keys": "Return" }
        }

        or

        {
        "thought": "Scrolling is needed to see more options.",
        "action": { "type": "scroll", "clicks": 2 }
        }

        or

        {
        "thought": "Instruction completed.",
        "action": { "type": "done" }
        }

        or

        {
        "thought": "It is not possible to continue (describe why).",
        "action": { "type": "fail", "reason": "Describe obstacle or limitation." }
        }

        - For "click", give x and y as normalized coordinates (; top-left = 0,0, bottom-right = 1,1).[1]
        - For "type", "key", "scroll", "done", or "fail", fill in only relevant fields.
        - Do NOT output anything except the single, machine-parseable JSON object above.

        Best Practices:
        - Always describe which UI element you are targeting and why.
        - If the previous action failed, explain exactly why, and try a different strategy (scroll, use keyboard, click elsewhere, etc.)
        - If a previous action did not work, explain your new strategy and never repeat the identical action more than twice.
        - Before typing, click to focus the corresponding field.
        - As soon as the instruction is achieved, use the "done" action.
        - If truly stuck, use "fail" and explain the reason.
        - Detect dialogs, pop-ups, error messages, or loading screens and factor them into your reasoning for every action.

        Example responses:

        {
        "thought": "I see a web browser window; the address bar is at the top. I will click the address bar to enter a URL.",
        "action": { "type": "click", "x": 0.12, "y": 0.07 }
        }

        {
        "thought": "The address bar is focused. I will type the website URL as instructed.",
        "action": { "type": "type", "text": "https://en.wikipedia.org" }
        }

        {
        "thought": "The address is typed; pressing Enter to go to the site.",
        "action": { "type": "key", "keys": "Return" }
        }

        {
        "thought": "The target website is now loaded. Task complete.",
        "action": { "type": "done" }
        }
        '''



    def _build_user_prompt(self, task_instruction, screenshot):
        """Build the user prompt with task instruction and action history."""
        # Derive (h, w) just to mention the true size; we still work in normalized coords.
        if isinstance(screenshot, np.ndarray):
            h, w = screenshot.shape[0], screenshot.shape[1]
        elif isinstance(screenshot, Image.Image):
            w, h = screenshot.size
        else:
            w, h = 1920, 1080

        prompt = f"Task Instruction: {task_instruction}\n\n"

        if self.action_history:
            prompt += "Previous Actions (most recent last):\n"
            for i, action in enumerate(self.action_history[-self.max_history:], 1):
                prompt += f"{i}. {action}\n"
            prompt += "\n"

        if h is not None and w is not None:
            prompt += f"Screenshot resolution: {w}x{h} pixels.\n"
        prompt += (
            "Remember: choose click coordinates in normalized space [0,1]x[0,1]. "
            "Top-left is (0,0), Bottom-right is (1,1).\n\n"
        )
        prompt += "Based on the current screenshot and the task instruction, what is the next best action?\n"
        prompt += "Respond with ONLY a JSON object containing 'thought' and 'action'."

        return prompt

    def _convert_action_obj_to_string(self, action_obj, screen_w, screen_h):
        """
        Convert structured action to executable pyautogui string.
        Handles coordinate mapping (Normalized -> Pixels) and string escaping.
        """
        if not isinstance(action_obj, dict):
            return "FAIL"

        # 1. Extract Type
        atype = action_obj.get("type", "").lower()

        # 2. Setup Defaults for Safety
        if not screen_w or not screen_h:
            screen_w, screen_h = 1920, 1080 

        # --- ACTION HANDLERS ---

        if atype == "click":
            # Get normalized coordinates (0.0 to 1.0)
            x_norm = action_obj.get("x")
            y_norm = action_obj.get("y")

            # Validate and clamp normalized coordinates
            try:
                x_norm = float(x_norm)
            except Exception:
                x_norm = 0.5
            try:
                y_norm = float(y_norm)
            except Exception:
                y_norm = 0.5

            # Clamp to [0.0, 1.0]
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            # CONVERT TO PIXELS (use int centers)
            px = int(round(x_norm * (screen_w - 1)))
            py = int(round(y_norm * (screen_h - 1)))

            # Generate OSWorld-safe string
            # Move first, then click. This updates hover states correctly.
            # Use explicit coordinates when clicking to avoid relying on current mouse
            # position in environments where moveTo may be less reliable.
            return f"pyautogui.moveTo({px}, {py}); pyautogui.click({px}, {py})"

        elif atype == "type":
            text = action_obj.get("text", "")
            if not isinstance(text, str):
                return "FAIL"
            
            # Escape single quotes and backslashes to prevent Python syntax errors
            # Example: "I'm here" -> "I\\'m here"
            safe_text = text.replace("\\", "\\\\").replace("'", "\\'")
            
            # Interval helps responsiveness in VMs
            return f"pyautogui.write('{safe_text}', interval=0.05)"

        elif atype == "key":
            keys = action_obj.get("keys", "")
            if not keys:
                return "FAIL"
            
            # Sanitize input
            safe_keys = keys.replace("'", "").replace('"', '')
            
            # Handle special mapping if needed, otherwise pass through
            return f"pyautogui.press('{safe_keys}')"

        elif atype == "scroll":
            clicks = action_obj.get("clicks", 0)
            try:
                clicks_int = int(clicks)
                # Negate because typically 'scroll down' is negative in many APIs, 
                # but usually agents predict positive for "scroll down". 
                # Adjust sign based on your specific OSWorld setup.
                # Here we assume agent says "2" means "scroll down 2 units".
                return f"pyautogui.scroll({clicks_int * -100})" 
            except Exception:
                return "FAIL"

        elif atype == "done":
            return "DONE"

        elif atype == "fail":
            return "FAIL"

        else:
            return "FAIL"
    def _detect_loop_and_maybe_fail(self, new_action_str):
        """
        Simple loop breaker: if the same action has been repeated many times,
        force FAIL to avoid infinite loops.
        """
        # Check last 3 actions
        if len(self.action_history) >= 3:
            last3 = self.action_history[-3:]
            if all(a == new_action_str for a in last3):
                print("Detected repeated action loop (same action 3+ times). Forcing FAIL.")
                return "FAIL"
        return new_action_str

    def act(self, observation, task_instruction):
        """
        Decide the next action based on the current observation.

        Args:
            observation: OSWorld observation dict with 'screenshot' key
            task_instruction: String describing the task to complete

        Returns:
            Action string in OSWorld format (e.g., "click(500, 300)" or "DONE")
        """
        self.step_count += 1

        screenshot = observation.get("screenshot")
        #pil_image = self._save_debug_screenshot(screenshot, self.step_count)
        if screenshot is None:
            print("ERROR: No screenshot in observation")
            return "FAIL"
        # Encode screenshot to base64
        try:
            base64_image = self._encode_image(screenshot)
        except Exception as e:
            print(f"ERROR: Failed to encode screenshot: {e}")
            return "FAIL"

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task_instruction, screenshot)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            },
        ]

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.1,  # lower for more deterministic control
            )

            raw_response = response.choices[0].message.content.strip()
            print(f"\n--- Agent Response (Step {self.step_count}) ---")
            print(raw_response)

            # Strip markdown code fences if the model ignores instructions
            if raw_response.startswith("```"):
                lines = raw_response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                raw_response = "\n".join(json_lines).strip()

            action_data = json.loads(raw_response)

            thought = action_data.get("thought", "")
            action_obj = action_data.get("action", {})

            print(f"Thought: {thought}")
            print(f"Action object: {action_obj}")
            # Determine screen/image size from the provided screenshot instead of
            # relying on local `pyautogui.size()` which returns the host display
            # size (the Mac) and may not match the VM / screenshot dimensions.
            if isinstance(screenshot, np.ndarray):
                # numpy array shape: (H, W, C) or (H, W)
                screen_h, screen_w = screenshot.shape[0], screenshot.shape[1]
            elif isinstance(screenshot, Image.Image):
                screen_w, screen_h = screenshot.size
            else:
                # Fallback to a reasonable default
                screen_w, screen_h = 1920, 1080

            action_str = self._convert_action_obj_to_string(action_obj, screen_w, screen_h)
            action_str = self._detect_loop_and_maybe_fail(action_str)

            self.action_history.append(action_str)
            return action_str

        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON response: {e}")
            print(f"Raw response: {raw_response}")
            return "FAIL"
        except Exception as e:
            print(f"ERROR: API call failed: {e}")
            return "FAIL"

    def reset(self):
        """Reset the agent's state for a new task."""
        self.action_history = []
        self.step_count = 0
