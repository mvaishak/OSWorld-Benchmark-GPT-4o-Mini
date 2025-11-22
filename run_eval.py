# import sys
# import os
# import time
# import json
# import logging
# import argparse
# from datetime import datetime
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # --- DYNAMIC PATH HANDLING ---
# # Checks .env first, then defaults to a hardcoded path or None
# OSWORLD_PATH = os.getenv("OSWORLD_ROOT")
# if OSWORLD_PATH and OSWORLD_PATH not in sys.path:
#     sys.path.append(OSWORLD_PATH)
# # ----------------------------------

# from environment_setup import setup_environment
# from agent import GPT40MiniAgent
# from task_loader import TaskLoader

# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("osworld_eval.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("MainLoop")

# def save_artifact(run_dir, task_id, step_idx, screenshot, action, thought):
#     step_dir = os.path.join(run_dir, task_id)
#     os.makedirs(step_dir, exist_ok=True)
    
#     with open(os.path.join(step_dir, f"step_{step_idx:02d}.png"), "wb") as f:
#         f.write(screenshot)
        
#     trace_data = {"step": step_idx, "thought": thought, "action": action}
#     with open(os.path.join(step_dir, f"step_{step_idx:02d}_trace.json"), "w") as f:
#         json.dump(trace_data, f, indent=2)


# def run_evaluation(args):
#     if not args.vm_path or not os.path.exists(args.vm_path):
#         logger.error(f"VM Path invalid: {args.vm_path}")
#         return

#     env = setup_environment(args)
#     agent = GPT40MiniAgent(api_key=args.api_key)
#     loader = TaskLoader(registry_path=args.registry, selection_path=args.selection)
#     tasks = loader.get_tasks()

#     if not tasks:
#         logger.error("No tasks loaded.")
#         if env: env.close()
#         return

#     # Create run-specific directory
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = os.path.join(args.output_dir, f"run_{run_id}")
#     os.makedirs(results_dir, exist_ok=True)

#     try:
#         for task in tasks:
#             task_id = task.get('id', 'unknown')
#             domain = task.get('domain', 'unknown')
#             instruction = task.get('instruction', 'No instruction')
            
#             logger.info(f"Starting Task {task_id} ({domain})")
            
#             # Create a folder for this specific task
#             task_dir = os.path.join(results_dir, domain, task_id)
#             os.makedirs(task_dir, exist_ok=True)

#             obs = env.reset(task_config=task)
#             time.sleep(2) 
            
#             # --- NEW: Accumulate trajectory in memory ---
#             trajectory = []
#             success_score = 0.0
            
#             for step_idx in range(args.max_steps):
#                 # 1. Save Screenshot (Keep these separate for visual debugging)
#                 screenshot_path = os.path.join(task_dir, f"step_{step_idx:02d}.png")
#                 with open(screenshot_path, "wb") as f:
#                     f.write(obs['screenshot'])

#                 # 2. Agent Inference
#                 response, trace_log = agent.predict(instruction, obs, trajectory)
#                 action = response.get('action')
#                 thought = response.get('thought')

#                 # 3. Record Step Data
#                 step_data = {
#                     "step_id": step_idx,
#                     "thought": thought,
#                     "action": action,
#                     "screenshot_file": f"step_{step_idx:02d}.png",
#                     "response_raw": trace_log
#                 }
#                 trajectory.append(step_data)

#                 # 4. Execute
#                 obs, reward, done, info = env.step(action)
                
#                 if done:
#                     logger.info(f"Task {task_id} signalled DONE.")
#                     break
            
#             # 5. Evaluate
#             try:
#                 success_score = env.evaluate()
#                 logger.info(f"Task {task_id} Score: {success_score}")
#             except Exception as e:
#                 logger.error(f"Evaluation failed for {task_id}: {e}")
#                 success_score = 0.0

#             # --- NEW: Save Single Trajectory File ---
#             traj_file = os.path.join(task_dir, "trajectory.json")
#             final_log = {
#                 "task_id": task_id,
#                 "domain": domain,
#                 "instruction": instruction,
#                 "success_score": success_score,
#                 "steps_taken": len(trajectory),
#                 "steps": trajectory # List of all steps
#             }
#             with open(traj_file, "w") as f:
#                 json.dump(final_log, f, indent=2)
            
#             # Save simple score file for quick reading
#             with open(os.path.join(task_dir, "score.txt"), "w") as f:
#                 f.write(str(success_score))

#     finally:
#         if env:
#             env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="OSWorld-Mini-Eval Executor")
    
#     # ARGUMENTS WITH .ENV DEFAULTS
#     # If the flag is NOT provided in CLI, it takes the value from os.getenv
    
#     parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), 
#                         help="OpenAI API Key (default: env OPENAI_API_KEY)")
    
#     parser.add_argument("--registry", default=os.getenv("OSWORLD_REGISTRY"), 
#                         help="Path to full test_all.json (default: env OSWORLD_REGISTRY)")
    
#     parser.add_argument("--vm_path", default=os.getenv("OSWORLD_VM_PATH"), 
#                         help="Path to VMware .vmx (default: env OSWORLD_VM_PATH)")
    
#     parser.add_argument("--output_dir", default=os.getenv("OSWORLD_OUTPUT_DIR", "./results"), 
#                         help="Artifacts directory")
    
#     parser.add_argument("--selection", required=False, 
#                         help="Path to specific task ID map JSON")
    
#     parser.add_argument("--max_steps", type=int, default=15, help="Max steps per task")
#     parser.add_argument("--provider", default="vmware", choices=["vmware"]) 
#     parser.add_argument("--headless", action="store_true")

#     args = parser.parse_args()
    
#     # Check for missing required args (if not in env and not in CLI)
#     if not args.api_key:
#         parser.error("--api_key is required (or set OPENAI_API_KEY in .env)")
#     if not args.registry:
#         parser.error("--registry is required (or set OSWORLD_REGISTRY in .env)")
#     if not args.vm_path:
#         parser.error("--vm_path is required (or set OSWORLD_VM_PATH in .env)")

#     run_evaluation(args)
import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OSWORLD_PATH = os.getenv("OSWORLD_ROOT")
if OSWORLD_PATH and OSWORLD_PATH not in sys.path:
    sys.path.append(OSWORLD_PATH)

from environment_setup import setup_environment
from agent import GPT40MiniAgent
from task_loader import TaskLoader

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("osworld_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MainLoop")

def determine_status(score, step_count, max_steps, final_action, error_msg=None):
    """
    Heuristic to diagnose the run outcome.
    """
    if error_msg:
        return "ERROR", error_msg

    if score == 1.0:
        return "SUCCESS", "Task Completed Successfully"

    # Failure Analysis
    if "FAIL" in str(final_action):
        return "FAILURE", "Agent Gave Up (Self-Reported)"
    
    if step_count >= max_steps:
        return "FAILURE", "Timeout (Max Steps Reached)"
    
    if "DONE" in str(final_action):
        return "FAILURE", "Wrong Result (Agent claimed success but eval failed)"

    return "FAILURE", "Unknown (Loop ended prematurely)"

def run_evaluation(args):
    if not args.vm_path or not os.path.exists(args.vm_path):
        logger.error(f"VM Path invalid: {args.vm_path}")
        return

    env = setup_environment(args)
    agent = GPT40MiniAgent(api_key=args.api_key)
    loader = TaskLoader(registry_path=args.registry, selection_path=args.selection)
    tasks = loader.get_tasks()

    if not tasks:
        logger.error("No tasks loaded.")
        if env: env.close()
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(results_dir, exist_ok=True)

    # Store summary for final report
    run_summary = []

    try:
        for task in tasks:
            task_id = task.get('id', 'unknown')
            domain = task.get('snapshot', 'unknown')
            instruction = task.get('instruction', 'No instruction')
            
            logger.info(f"--- Starting Task {task_id} ({domain}) ---")
            
            task_dir = os.path.join(results_dir, domain, task_id)
            os.makedirs(task_dir, exist_ok=True)

            obs = env.reset(task_config=task)
            time.sleep(2) 
            
            trajectory = []
            success_score = 0.0
            final_action = ""
            eval_error = None
            
            step_idx = 0
            for step_idx in range(args.max_steps):
                # 1. Save Screenshot
                screenshot_path = os.path.join(task_dir, f"step_{step_idx:02d}.png")
                with open(screenshot_path, "wb") as f:
                    f.write(obs['screenshot'])

                # 2. Agent Inference
                response, trace_log = agent.predict(instruction, obs, trajectory)
                action = response.get('action')
                thought = response.get('thought')
                final_action = action # Track last action for diagnosis

                # 3. Record Step
                step_data = {
                    "step_id": step_idx,
                    "thought": thought,
                    "action": action,
                    "screenshot_file": f"step_{step_idx:02d}.png",
                    "response_raw": trace_log
                }
                trajectory.append(step_data)

                # 4. Execute
                obs, reward, done, info = env.step(action)
                
                if done:
                    logger.info(f"Task {task_id} signalled DONE.")
                    break
            
            # 5. Evaluate
            try:
                success_score = env.evaluate()
                logger.info(f"Task {task_id} Raw Score: {success_score}")
            except Exception as e:
                logger.error(f"Evaluation script failed for {task_id}: {e}")
                success_score = 0.0
                eval_error = str(e)

            # 6. Diagnose Outcome
            # We pass step_idx + 1 because range is 0-indexed
            status, reason = determine_status(success_score, step_idx + 1, args.max_steps, final_action, eval_error)
            
            logger.info(f"Task Outcome: [{status}] - Reason: {reason}")

            # 7. Save Enhanced Trajectory
            traj_file = os.path.join(task_dir, "trajectory.json")
            final_log = {
                "task_id": task_id,
                "domain": domain,
                "instruction": instruction,
                "status": status,           # NEW: explicit status
                "failure_reason": reason,   # NEW: explicit reason
                "success_score": success_score,
                "steps_taken": len(trajectory),
                "steps": trajectory
            }
            with open(traj_file, "w") as f:
                json.dump(final_log, f, indent=2)
            
            # Add to summary list
            run_summary.append({
                "id": task_id,
                "domain": domain,
                "status": status,
                "reason": reason,
                "steps": len(trajectory)
            })

    finally:
        if env:
            env.close()
        
        # --- FINAL REPORT ---
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY (Run ID: {run_id})")
        print("="*60)
        print(f"{'Domain':<15} | {'Task ID':<10} | {'Status':<8} | {'Reason'}")
        print("-" * 60)
        for item in run_summary:
            # Truncate reason for cleaner table
            reason_short = (item['reason'][:30] + '..') if len(item['reason']) > 30 else item['reason']
            print(f"{item['domain']:<15} | {item['id'][:8]:<10} | {item['status']:<8} | {reason_short}")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSWorld-Mini-Eval Executor")
    
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--registry", default=os.getenv("OSWORLD_REGISTRY"))
    parser.add_argument("--vm_path", default=os.getenv("OSWORLD_VM_PATH"))
    parser.add_argument("--output_dir", default=os.getenv("OSWORLD_OUTPUT_DIR", "./results"))
    parser.add_argument("--selection", required=False)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--provider", default="vmware", choices=["vmware"]) 
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()
    
    if not args.api_key: parser.error("--api_key is required")
    if not args.registry: parser.error("--registry is required")
    if not args.vm_path: parser.error("--vm_path is required")

    run_evaluation(args)