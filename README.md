# OSWorld Benchmark with GPT-4o-Mini

This repository contains a lightweight evaluation script for running agents on the [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark, specifically optimized for **GPT-4o-Mini**.

## Overview

The project implements a custom agent loop that interacts with the OSWorld environment (running on VMware) to perform computer control tasks. It includes:
- **`agent.py`**: A `pyautogui`-based agent that uses GPT-4o-Mini for vision-based decision making.
- **`run_eval.py`**: The main execution script that loads tasks, runs the agent, and evaluates performance.
- **`environment_setup.py`**: Handles the initialization of the OSWorld `DesktopEnv` with VMware support.

## Prerequisites

1.  **OSWorld Environment**: You must have the OSWorld repository cloned and set up.
2.  **VMware Fusion/Workstation**: Required for running the OSWorld VM.
3.  **Python 3.10+**: Recommended.

## Setup

1.  **Clone this repository**.
2.  **Install dependencies**:
    ```bash
    pip install openai python-dotenv pyautogui
    # Ensure you have the OSWorld dependencies installed as well
    ```
3.  **Configure Environment Variables**:
    Create a `.env` file in the root directory with the following:
    ```env
    OPENAI_API_KEY=sk-...
    OSWORLD_ROOT=/path/to/OSWorld
    OSWORLD_VM_PATH=/path/to/Ubuntu/Ubuntu.vmx
    OSWORLD_REGISTRY=/path/to/OSWorld/evaluation_examples/
    ```

## Usage

To run the evaluation on a specific set of tasks defined in a selection file (e.g., `tests.json`):

```bash
python run_eval.py --selection tests.json --max_steps 15
```

### Arguments

- `--selection`: Path to a JSON file mapping domains to task IDs (see `tests.json` for example).
- `--max_steps`: Maximum number of steps allowed per task (default: 15).
- `--headless`: Run without the visible VMware GUI (if supported).

## Project Structure

- `agent.py`: Core agent logic and prompt construction.
- `run_eval.py`: Main entry point for running evaluations.
- `environment_setup.py`: OSWorld environment configuration.
- `task_loader.py`: Helper to load task configurations from the OSWorld registry.
- `reports/`: Contains analysis notebooks and reports.
