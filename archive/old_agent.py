
class GPT40MiniAgent:
    """
    The cognitive core of the evaluation script, designed to interface with GPT-4o-mini
    within the OSWorld environment[cite: 104].
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # Screen resolution is critical for grounding (Section 5.1) [cite: 114]
        self.screen_width = 1920
        self.screen_height = 1080
        self.system_prompt = self._construct_system_prompt()

    def _construct_system_prompt(self) -> str:
        """
        Constructs the rigid system prompt required to ground the 'mini' model.
        Definitions based on Section 5.1 and Table 1[cite: 107, 108, 110, 48].
        """
        return f"""You are a precise computer control agent. [cite: 111]
You interact with a computer running at {self.screen_width}x{self.screen_height} resolution (top-left is 0,0). [cite: 114]

AVAILABLE ACTION SPACE (Python syntax):
1. click(x, y): Left click at coordinates.
2. right_click(x, y): Right click at coordinates.
3. double_click(x, y): Double click at coordinates.
4. drag(start_x, start_y, end_x, end_y): Click-and-hold from start to end.
5. type(text): Type string into active field.
6. press(key): Press key (e.g., 'enter', 'tab', 'esc').
7. hotkey(key1, key2): Key combo (e.g., 'ctrl', 'c').
8. WAIT: Pause execution for UI rendering.
9. FAIL: Signal task is impossible.
10. DONE: Signal successful task completion. [cite: 48, 49, 50]

OUTPUT FORMAT:
You must output VALID JSON only. No conversational text.
{{
  "thought": "Reasoning about the current state and next step...",
  "action": "click(500, 300)"
}} 
"""

    def predict(self, instruction: str, observation: Dict[str, Any], history: List[Dict]) -> Tuple[Dict, str]:
        """
        Processes the hybrid observation (Screenshot + A11y Tree) and generates an action.
        Implements logic from Section 5.2[cite: 117].
        """
        # 1. Visual Processing: Encode screenshot [cite: 118]
        # Assuming observation['screenshot'] is raw bytes or path; converting to base64
        base64_image = self._encode_image(observation['screenshot'])

        # 2. Textual Processing: Parse A11y tree [cite: 119, 120]
        # The A11y tree helps mitigate hallucination of click locations [cite: 42]
        a11y_tree = observation.get('accessibility_tree', 'No accessibility info available.')

        # 3. Construct Multimodal Prompt [cite: 121, 122]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task: {instruction}\n"
                                f"Accessibility Tree: {a11y_tree}\n"
                                f"History: {json.dumps(history)}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"  # Maintain high res for text legibility [cite: 118]
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=300, # Limit generation for speed
                temperature=0.0 # Deterministic output
            )
            raw_content = response.choices[0].message.content
            logger.debug(f"Agent Trace: {raw_content}") [cite: 75]
            
            return self._parse_response(raw_content)

        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return {"action": "WAIT"}, "API_ERROR"

    def _parse_response(self, content: str) -> Tuple[Dict, str]:
        """
        Parses and validates the agent's raw response.
        Implements the 'safety valve' logic from Section 5.3[cite: 123].
        """
        try:
            # Regex to extract JSON object, handling potential conversational filler [cite: 124]
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")
            
            response_json = json.loads(match.group(0))
            
            # Validate Action Schema
            action_str = response_json.get('action', '')
            if not self._validate_action(action_str):
                 # Logic from Section 5.3: Log hallucination and issue WAIT 
                logger.warning(f"Hallucination detected: Invalid action signature '{action_str}'")
                return {"action": "WAIT"}, content

            return response_json, content

        except Exception as e:
            logger.error(f"Parse Error: {str(e)}")
            return {"action": "WAIT"}, content

    def _validate_action(self, action_str: str) -> bool:
        """
        Validates that the action exists in the allowed list and parameters are sane.
        [cite: 126]
        """
        valid_primitives = [
            "click", "right_click", "double_click", "drag", 
            "type", "press", "hotkey", "WAIT", "DONE", "FAIL"
        ]
        
        # Simple check: does the string start with a valid primitive?
        # A robust implementation would use AST parsing to verify arguments (integers vs strings)
        is_valid = any(action_str.startswith(prim) for prim in valid_primitives)
        return is_valid

    def _encode_image(self, image_source) -> str:
        """Helper to handle image encoding (path or bytes)."""
        if isinstance(image_source, str) and os.path.exists(image_source):
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_source, bytes):
             return base64.b64encode(image_source).decode('utf-8')
        return ""
