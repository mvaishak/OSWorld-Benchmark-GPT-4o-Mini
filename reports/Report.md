# Evaluation Report: Feasibility of Efficient Vision-Language Models for Desktop Automation

- **Author:** Vaishak Menon
- **Date:** November 21, 2025
- **Subject:** Performance Analysis of GPT-4o-mini on OSWorld Benchmark (Subset)

## 1. Executive Summary

I evaluated the `gpt-4o-mini` model on a subset of 10 diverse tasks from the OSWorld benchmark to determine if cost-effective "mini" class models can serve as viable desktop agents. My evaluation reveals a **0.0% success rate** across all domains (Chrome, LibreOffice, OS, VS Code), with an average run length of 6.0 steps.

While the model demonstrated strong **semantic understanding** (correctly identifying *what* needed to be done, such as "Open File Menu"), it consistently failed at **execution grounding** (accurately predicting *where* to click). This report analyzes these failures and proposes a strategic roadmap for improving performance through architectural changes rather than model scaling alone.

I have experimented with different strategies but for the sake of time and simplicity I have gone ahead with the one the approach that more closely aligned with the instructions that were shared. I did work with accessibility tree initially but I found it to be a bit complex and not as effective as I had hoped with the current model. 

The archive folder contains a couple of alternative approaches that I had tried but did not work out.

## 2. Failure Mode Analysis

My analysis of the trajectory logs identified three primary distinct failure patterns that explain the 0% success rate:

### A. The "Hallucination of Success" (Wrong Result - 40%)
In 40% of cases, the agent declared the task `DONE` prematurely.
* **Observation:** The agent would perform a logical action (e.g., clicking "Save"), assume the action succeeded without visual verification, and terminate.
* **Root Cause:** The model lacks a "closed-loop" verification step. It treats the output of an action (the click) as proof of the outcome (the save), failing to verify if the state of the environment actually changed.

### B. The "Grounding Gap" (Agent Gave Up - 30%)
In 30% of tasks, the agent explicitly outputted `FAIL` after failing to locate UI elements.
* **Observation:** Instructions often referenced elements like "The search bar" or "The extensions icon." The vision-only model struggled to distinguish these small, low-contrast elements on a 1080p screen.
* **Root Cause:** "Mini" models have lower visual acuity and struggle with the "Vision-to-Coordinate" translation task. They often hallucinated coordinates or clicked empty space nearby.

### C. The "Looping Trap" (Timeout - 30%)
In the remaining 30%, the agent exhausted the 15-step limit.
* **Observation:** I observed the agent repeatedly attempting the same failed action (e.g., clicking a menu that wouldn't open) without changing its strategy.
* **Root Cause:** A lack of short-term "episodic memory." The agent's context window contains the history of commands, but it failed to synthesize that history into a realization that "Strategy A is failing, I must switch to Strategy B."

---

## 3. Recommendations for Performance Improvement

Based on these findings, I propose three specific avenues for improvement: Advanced Prompting, Tool Integration, and UI-Specific Optimization.

### I. Better Prompting Strategies

Standard instruction prompting proved insufficient. I recommend implementing **Set-of-Mark (SoM)** and **Chain-of-Thought (CoT)** prompting to offload spatial reasoning.

* **Visual Grounding via Set-of-Mark (SoM):**
    Instead of asking the model to predict x/y coordinates (a continuous regression task it is poor at), I propose overlaying a grid of numbered bounding boxes on the screenshot *before* it reaches the model.
    * *Impact:* This converts the task from "Guess the coordinates of the File menu" (Hard) to "Read the number on the File menu" (Easy). This would directly mitigate the 30% "Agent Gave Up" rate caused by grounding failures.

* **Enforced Chain-of-Thought (CoT):**
    The logs showed the agent rushing to execute actions. I propose modifying the system prompt to enforce a structured thinking block *before* the JSON action block.
    * *Structure:* `{"Observation": "The menu did not open.", "Hypothesis": "My click was likely off-center.", "Plan": "I will try the keyboard shortcut 'Alt+F' instead.", "Action": ...}`
    * *Impact:* This forces the model to perform the self-correction it currently lacks, potentially reducing the "Timeout" failures caused by repetitive looping.

### II. Tool Integration: Memory & Reflexion

To solve the "Hallucination of Success," the agent needs tools that provide objective feedback about the environment state.

* **Visual State Hashing (Reflexion):**
    I recommend implementing a "Reflexion Tool" that compares the perceptual hash (pHash) of the screenshot before and after an action.
    * *Mechanism:* If `hash(screen_t) == hash(screen_t+1)`, the system should inject a prompt: *"System Notice: The screen state did not change. Your last action had no effect."*
    * *Impact:* This provides the "closed-loop" feedback necessary to prevent the agent from hallucinating success or getting stuck in loops.

* **Separation of Concerns (Planner-Grounder Architecture):**
    Currently, one model does everything. I propose splitting the agent into two distinct modules:
    1.  **Planner (LLM):** Decides *what* to do (e.g., "Click the Save Button").
    2.  **Grounder (Heuristic/Tool):** A specialized Python script or smaller model that takes the text "Save Button," searches the Accessibility Tree, and calculates the exact center coordinates.
    * *Impact:* This removes the burden of pixel-perfect accuracy from the LLM, playing to the strengths of code-based tools.

### III. UI-Specific Hints & Environment Optimization

The evaluation highlighted environmental factors that significantly degraded performance.

* **Accessibility Tree "Smart Filtering":**
    My logs showed that raw Accessibility Trees (A11y) often exceeded the context window (e.g., VS Code logs were >200k tokens). Simply truncating them blindly removes critical data.
    * *Solution:* Implement a heuristic filter that prioritizes "interactive" nodes (buttons, links, inputs) and discards static containers (divs, panes) before passing the tree to the model.

* **Coordinate Space Normalization (Mac/VM Mismatch):**
    Running the evaluation on a macOS host introduced a discrepancy between the host's high-DPI (Retina) coordinates and the VM's standard 1080p resolution.
    * *Solution:* I must implement a coordinate normalization layer in the `DesktopEnv` wrapper. This layer would automatically scale the agent's output coordinates by the detected DPI ratio (e.g., dividing by 2.0) to ensure clicks land accurately on the VM's UI elements.

## 4. Conclusion

My evaluation demonstrates that `gpt-4o-mini` cannot currently function as a reliable desktop agent using a naive "screenshot-to-action" approach. The 0% success rate is driven not by a lack of reasoning capability, but by the inability to reliably ground that reasoning in the visual interface.

However, the failures are systematic rather than random. By implementing **Set-of-Mark prompting** to solve grounding, **Reflexion tools** to solve looping, and **Coordinate Normalization** to fix environmental mismatches, I believe it is possible to build a functional agent on this cost-effective architecture.
