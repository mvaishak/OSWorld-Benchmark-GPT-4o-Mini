import json
import logging
from typing import List, Dict, Any
import sys
import os
sys.path.append("/Users/mvaishak/Developer/AIAgents/OSWorld")

logger = logging.getLogger("TaskLoader")

class TaskLoader:
    """
    Loads tasks by using the domain keys from the selection map to locate
    specific JSON files in the OSWorld examples directory.
    """

    def __init__(self, registry_path: str, selection_path: str):
        """
        Args:
            registry_path: Path to the root 'evaluation_examples' folder.
            selection_path: Path to your selection map JSON.
        """
        self.registry_path = registry_path
        self.selection_path = selection_path

    def get_tasks(self) -> List[Dict[str, Any]]:
        # 1. Load the selection map
        if not self.selection_path or not os.path.exists(self.selection_path):
            logger.error(f"Selection file not found: {self.selection_path}")
            return []

        try:
            with open(self.selection_path, 'r') as f:
                selection_map = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in selection file: {self.selection_path}")
            return []

        loaded_tasks = []

        # 2. Iterate through Domain -> IDs
        for domain, task_ids in selection_map.items():
            # Construct path: registry_path + "examples" + domain
            # e.g., /.../evaluation_examples/examples/chrome
            domain_dir = os.path.join(self.registry_path, "examples", domain)
            
            if not os.path.exists(domain_dir):
                logger.warning(f"Domain directory not found: {domain_dir}")
                continue

            for task_id in task_ids:
                # Construct specific file path: UUID.json
                task_file = os.path.join(domain_dir, f"{task_id}.json")
                
                if os.path.exists(task_file):
                    try:
                        with open(task_file, 'r') as f:
                            task_data = json.load(f)
                            #task_data['domain'] = domain
                            # Some example files might be a list [{}], others just a dict {}
                            if isinstance(task_data, list):
                                loaded_tasks.extend(task_data)
                            else:
                                loaded_tasks.append(task_data)
                    except Exception as e:
                        logger.error(f"Failed to load JSON for {task_id}: {e}")
                else:
                    logger.warning(f"Task file not found: {task_file}")

        logger.info(f"Successfully loaded {len(loaded_tasks)} tasks from {self.registry_path}")
        return loaded_tasks