import base64
import json
from io import BytesIO

from openai import OpenAI
from PIL import Image
import numpy as np


class GPT4oMiniAgent:
    """
    Improved GPT-4o-mini computer-use agent for OSWorld.
    
    This agent:
    - Takes OSWorld observations (screenshots + task instruction)
    - Calls GPT-4o-mini with vision capabilities
    - Returns executable action strings in OSWorld's pyautogui format
    """

    def __init__(self, api_key, model="gpt-4o-mini", max_history=5):
        """
        Initialize the agent.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            max_history: Maximum number of previous actions to keep in context
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_history = max_history
        self.action_history = []
        self.step_count = 0

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
        return """You are an expert computer-use agent controlling a linux desktop GUI in the OSWorld environment.

Your task is to complete user instructions by interacting with the desktop interface through mouse clicks, keyboard input, and other actions.

**Available Actions:**
1. **click(x, y)** - Click at coordinates (x, y) where x and y are integers from 0-1000
   - (0, 0) is top-left corner
   - (1000, 1000) is bottom-right corner
   - Example: click(500, 300)

2. **type(text)** - Type the specified text
   - Example: type("Hello World")

3. **key(key_combination)** - Press a key or key combination
   - Examples: key("Return"), key("ctrl+c"), key("ctrl+v"), key("Tab")

4. **scroll(clicks)** - Scroll by the specified number of clicks (positive = down, negative = up)
   - Example: scroll(3) or scroll(-2)

5. **DONE** - Signal that the task is complete
   - Use this when you have successfully completed the instruction

6. **FAIL** - Signal that the task cannot be completed
   - Use this if you're stuck or the task is impossible

**Important Guidelines:**
- Analyze the screenshot carefully before deciding on an action
- Click on UI elements precisely (buttons, links, input fields, etc.)
- For text input, first click on the input field, then use type()
- Use keyboard shortcuts when appropriate (ctrl+c, ctrl+v, etc.)
- Be patient - some actions may take time to reflect in the UI
- If you see the expected result, use DONE
- If you're stuck in a loop or can't proceed, use FAIL

**Output Format:**
You must respond with ONLY a valid JSON object in this exact format:
{
  "thought": "Brief explanation of what you observe and why you're taking this action",
  "action": "the action string (e.g., 'click(500, 300)' or 'type(\"text\")' or 'DONE')"
}

Do NOT include any other text, markdown formatting, or code blocks. Output ONLY the JSON object."""

    def _build_user_prompt(self, task_instruction):
        """Build the user prompt with task instruction and action history."""
        prompt = f"**Task Instruction:** {task_instruction}\n\n"
        
        if self.action_history:
            prompt += "**Previous Actions:**\n"
            for i, action in enumerate(self.action_history[-self.max_history:], 1):
                prompt += f"{i}. {action}\n"
            prompt += "\n"
        
        prompt += "**Current Screenshot:** (see image below)\n\n"
        prompt += "Based on the current screenshot and task instruction, what is the next action to take?\n"
        prompt += "Remember: Output ONLY a JSON object with 'thought' and 'action' fields."
        
        return prompt

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
        
        # Extract screenshot
        screenshot = observation.get("screenshot")
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
        user_prompt = self._build_user_prompt(task_instruction)

        # Prepare messages for OpenAI API
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            raw_response = response.choices[0].message.content.strip()
            print(f"\n--- Agent Response (Step {self.step_count}) ---")
            print(raw_response)
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if raw_response.startswith("```"):
                # Extract JSON from code block
                lines = raw_response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or (not line.startswith("```")):
                        json_lines.append(line)
                raw_response = "\n".join(json_lines)
            
            action_data = json.loads(raw_response)
            
            # Extract thought and action
            thought = action_data.get("thought", "")
            action = action_data.get("action", "FAIL")
            
            print(f"Thought: {thought}")
            print(f"Action: {action}")
            
            # Store action in history
            self.action_history.append(action)
            
            return action

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
