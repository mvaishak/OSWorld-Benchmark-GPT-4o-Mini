import os
import base64
import json
import time
from openai import OpenAI
from io import BytesIO
from PIL import Image
import numpy as np


class GPT4oMiniAgent:
    def __init__(self, api_key, screen_width=1920, screen_height=1080):
        self.client = OpenAI(api_key=api_key)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Enhanced state management
        self.plan = []  # Step-by-step plan
        self.detailed_history = []  # Rich history with outcomes
        self.current_step_index = 0
        self.task_started = False
        self.current_task = ""
        self.last_action_data = None

    def _encode_image(self, image_data):
        """Converts a numpy array/PIL image to base64 string."""
        # Handle different input types from OSWorld
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, bytes):
            # Already bytes
            return base64.b64encode(image_data).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _get_planning_prompt(self):
        return """You are a computer-use agent capable of controlling an ubuntu GUI. 
You will receive a screenshot and a high-level task.

**Your Role: Task Planner**

Analyze the task and current screen state, then create a step-by-step plan.

**Output Format:**
You must output a JSON object with the following structure:
{
    "analysis": "Brief analysis of the current screen and what you see",
    "plan": [
        "Step 1: Detailed description of first action",
        "Step 2: Detailed description of second action",
        ...
    ],
    "potential_challenges": "Any UI elements that might need to be enabled or challenges to watch for"
}

**Important Guidelines:**
- Look carefully at the current UI state
- If you need to use UI elements (like bookmarks bar, toolbars), check if they're visible first
- If elements aren't visible, plan to enable them (usually via View menu, right-click, or settings)
- Break down complex tasks into small, concrete steps
- Consider common UI patterns: menus (File, Edit, View), settings (gear icon, three dots), right-click options
- ONLY output valid JSON, no additional text.
"""

    def _get_reflection_prompt(self):
        return """You are a computer-use agent capable of controlling a GUI.
You just took an action. Now reflect on what happened.

**Your Role: Action Evaluator**

Analyze the outcome of your last action based on the new screenshot.

**Output Format:**
You must output a JSON object with the following structure:
{
    "observation": "What do you see in the current screenshot? What changed?",
    "action_successful": true/false,
    "reason": "Why was the action successful or unsuccessful?",
    "next_step_adjustment": "Should we continue with the plan, or do we need to adjust? What should we do next?"
}

**Important Guidelines:**
- Compare what you see now vs what you expected
- Be honest about whether the action achieved its intended goal
- If something didn't work, think about why and what to try instead
- Look for error messages, unchanged UI, or unexpected states
- ONLY output valid JSON, no additional text.
"""

    def _get_action_prompt(self):
        return f"""You are a computer-use agent capable of controlling a GUI. 
You will receive a screenshot of the current desktop and a high-level task.

**Your Role: Action Executor**

Based on your plan and current state, decide the next action to take.

**Output Format:**
You must output a JSON object with the following structure:
{{
    "thought": "Brief reasoning about what to do next based on the UI and plan",
    "action_type": "click" | "type" | "scroll" | "key_press" | "done" | "fail",
    "parameters": {{
        "x": (int) horizontal pixel coordinate 0-{self.screen_width} (required for click),
        "y": (int) vertical pixel coordinate 0-{self.screen_height} (required for click),
        "text": (string) text to type (required for type),
        "key": (string) key name e.g. 'enter', 'ctrl+c' (required for key_press)
    }}
}}

**Important Screen Information:**
- The screenshot is {self.screen_width}x{self.screen_height} pixels
- Coordinates are in ABSOLUTE PIXELS: (0,0) is top-left, ({self.screen_width},{self.screen_height}) is bottom-right
- For example, the center of the screen is approximately ({self.screen_width//2},{self.screen_height//2})

**Constraints:**
- Look carefully at the screenshot to identify exact pixel locations of UI elements
- If the task is finished successfully, output action_type "done".
- If you are stuck or the task is impossible, output action_type "fail".
- Common UI patterns:
  * Three-dot menu (⋮) or hamburger menu (≡) often in top-right
  * Settings usually in gear icon or under menus
  * Right-click for context menus
  * View menu to show/hide toolbars and UI elements
- ONLY output valid JSON, no additional text.
"""

    def _call_llm(self, system_prompt, user_message, base64_image):
        """Helper method to call the LLM with consistent format."""
        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]

        try:
            response = self.client.responses.create(
                model="gpt-4o-mini",
                input=input_messages,
                instructions=system_prompt,
                temperature=0.7
            )
            
            raw = response.output_text.strip()

            # Remove triple backticks if present
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json\n", "").replace("json\r\n", "")

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {raw}")
            return None
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_plan(self, task_instruction, screenshot):
        """Generate initial plan for the task."""
        print("\n=== PLANNING PHASE ===")
        
        try:
            base64_image = self._encode_image(screenshot)
        except Exception as e:
            print(f"Error encoding image: {e}")
            return False

        user_message = f"""Task: {task_instruction}

Please analyze the current screen and create a detailed step-by-step plan to complete this task."""

        result = self._call_llm(self._get_planning_prompt(), user_message, base64_image)
        
        if result:
            print(f"Analysis: {result.get('analysis', 'N/A')}")
            print(f"Plan: {json.dumps(result.get('plan', []), indent=2)}")
            print(f"Potential Challenges: {result.get('potential_challenges', 'N/A')}")
            
            self.plan = result.get('plan', [])
            return True
        
        return False

    def _reflect(self, observation):
        """Reflect on the outcome of the last action."""
        if not self.last_action_data:
            return
        
        print("\n=== REFLECTION PHASE ===")
        
        screenshot = observation.get('screenshot')
        if screenshot is None:
            return
        
        try:
            base64_image = self._encode_image(screenshot)
        except Exception as e:
            print(f"Error encoding image: {e}")
            return

        user_message = f"""Last action taken: {self.last_action_data.get('action_type')} with parameters {self.last_action_data.get('parameters')}
Last thought: {self.last_action_data.get('thought')}

Please evaluate what happened after this action."""

        result = self._call_llm(self._get_reflection_prompt(), user_message, base64_image)
        
        if result:
            print(f"Observation: {result.get('observation', 'N/A')}")
            print(f"Action Successful: {result.get('action_successful', 'Unknown')}")
            print(f"Reason: {result.get('reason', 'N/A')}")
            print(f"Next Step Adjustment: {result.get('next_step_adjustment', 'N/A')}")
            
            # Add to detailed history
            self.detailed_history.append({
                'action': self.last_action_data,
                'reflection': result,
                'timestamp': time.time()
            })
            
            # Keep history manageable
            if len(self.detailed_history) > 15:
                self.detailed_history.pop(0)

    def _format_history_context(self):
        """Format history for context in action decision."""
        if not self.detailed_history:
            return "No previous actions yet."
        
        recent_history = self.detailed_history[-5:]  # Last 5 actions
        formatted = []
        
        for i, entry in enumerate(recent_history, 1):
            action = entry['action']
            reflection = entry['reflection']
            formatted.append(
                f"{i}. Action: {action.get('action_type')} - {action.get('thought')}\n"
                f"   Result: {'Success' if reflection.get('action_successful') else 'Failed'} - {reflection.get('reason')}"
            )
        
        return "\n".join(formatted)

    def act(self, observation, task_instruction):
        """Generate next action based on observation and task."""
        screenshot = observation.get('screenshot')
        
        if screenshot is None:
            print("Warning: No screenshot in observation")
            return "WAIT"
        
        # First time seeing this task - create a plan
        if not self.task_started or self.current_task != task_instruction:
            self.current_task = task_instruction
            self.task_started = True
            self.plan = []
            self.detailed_history = []
            self.current_step_index = 0
            
            if not self._create_plan(task_instruction, screenshot):
                print("Failed to create plan, proceeding without one")
        
        # Reflect on last action (if any)
        if self.last_action_data:
            self._reflect(observation)
        
        # Now decide next action
        print("\n=== ACTION PHASE ===")
        
        try:
            base64_image = self._encode_image(screenshot)
        except Exception as e:
            print(f"Error encoding image: {e}")
            return "WAIT"

        # Build context
        plan_context = "\n".join([f"{i+1}. {step}" for i, step in enumerate(self.plan)]) if self.plan else "No plan available"
        history_context = self._format_history_context()
        
        user_message = f"""Task: {task_instruction}

**Your Plan:**
{plan_context}

**Recent History:**
{history_context}

**Current Step:** You are working on step {self.current_step_index + 1} of {len(self.plan) if self.plan else 'unknown'}

Based on the current screenshot and your progress, what should you do next?"""

        result = self._call_llm(self._get_action_prompt(), user_message, base64_image)
        
        if result:
            print(f"Agent Thought: {result.get('thought', 'No thought')}")
            
            # Store for reflection
            self.last_action_data = result
            
            # Increment step if we're following the plan
            if result.get('action_type') not in ['done', 'fail']:
                self.current_step_index += 1
            
            # Convert to OSWorld action
            return self._parse_to_osworld_action(result)
        
        return "WAIT"

    def _parse_to_osworld_action(self, action_data):
        """Maps Agent JSON to OSWorld environment format."""
        atype = action_data.get("action_type")
        params = action_data.get("parameters", {})

        # Use absolute pixel coordinates directly
        if "x" in params and "y" in params:
            # Get coordinates as integers
            px = int(params.get('x', self.screen_width // 2))
            py = int(params.get('y', self.screen_height // 2))
            
            # Clamp to screen bounds
            px = max(0, min(self.screen_width - 1, px))
            py = max(0, min(self.screen_height - 1, py))
            
            # Debug logging
            print(f"[COORD DEBUG] Raw: ({params.get('x')}, {params.get('y')}) -> Pixel: ({px}, {py})")
        
        # OSWorld expects executable Python code strings
        if atype == "click":
            # Move to location first, then click for reliability
            return f"pyautogui.moveTo({px}, {py})\npyautogui.click()"
        
        elif atype == "type":
            text = params.get("text", "")
            # Properly escape the text
            escaped_text = text.replace("\\", "\\\\").replace("'", "\\'")
            return f"pyautogui.write('{escaped_text}', interval=0.05)"
        
        elif atype == "key_press":
            key = params.get("key", "")
            safe_key = key.replace("'", "").replace('"', '')
            return f"pyautogui.press('{safe_key}')"
        
        elif atype == "scroll":
            clicks = params.get("clicks", -10)
            return f"pyautogui.scroll({clicks})"
        
        elif atype == "done":
            return "DONE"
        
        elif atype == "fail":
            return "FAIL"
        
        else:
            print(f"Unknown action type: {atype}")
            return "WAIT"
