
def run_evaluation(args):
    # Validate critical paths before starting
    if not args.vm_path or not os.path.exists(args.vm_path):
        logger.error(f"VM Path invalid: {args.vm_path}")
        return

    # 1. Initialize Components
    env = setup_environment(args)
    agent = GPT40MiniAgent(api_key=args.api_key)
    loader = TaskLoader(registry_path=args.registry, selection_path=args.selection)
    
    # Create results directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(results_dir, exist_ok=True)

    tasks = loader.get_tasks()
    if not tasks:
        logger.error("No tasks found. Check your selection file or registry path.")
        return

    try:
        for task in tasks:
            task_id = task['id']
            instruction = task['instruction']
            logger.info(f"Starting Task {task_id}: {instruction}")
            
            obs = env.reset(task_config=task)
            time.sleep(2) # Stabilization Pause
            
            history = []
            
            for step_idx in range(args.max_steps):
                response, trace_log = agent.predict(instruction, obs, history)
                action = response.get('action')
                thought = response.get('thought')
                
                save_artifact(results_dir, task_id, step_idx, obs['screenshot'], action, thought)
                
                obs, reward, done, info = env.step(action)
                history.append({"step": step_idx, "action": action, "response": trace_log})
                
                if done:
                    logger.info(f"Task {task_id} signalled DONE.")
                    break
            
            try:
                score = env.evaluate()
                logger.info(f"Task {task_id} Score: {score}")
            except Exception as e:
                logger.error(f"Evaluation timed out for {task_id}: {e}")
                score = 0.0

            with open(os.path.join(results_dir, "results.csv"), "a") as f:
                f.write(f"{task_id},{score},{step_idx}\n")

    finally:
        if env:
            env.close()
            logger.info("Environment closed and resources released.")
