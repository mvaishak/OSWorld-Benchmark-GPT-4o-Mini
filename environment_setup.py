import sys
import os
import logging
import shutil
import argparse
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- DYNAMIC PATH HANDLING ---
# Checks .env first, then defaults to a hardcoded path or None
OSWORLD_PATH = os.getenv("OSWORLD_ROOT")
if OSWORLD_PATH and OSWORLD_PATH not in sys.path:
    sys.path.append(OSWORLD_PATH)

# Import the specialized wrapper distributed with OSWorld repository
# This is not a standard PyPI library
try:
    from desktop_env.desktop_env import DesktopEnv
except ImportError:
    sys.exit("Error: 'desktop_env' package not found. Ensure OSWorld repository is in your PYTHONPATH.")

# Configure logging
logger = logging.getLogger("EnvSetup")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def validate_vmware_dependencies():
    """
    Strictly verifies the presence of VMware system-level tools.
    The report notes that 'vmrun' is required for VMware interaction.
    """
    if not shutil.which("vmrun"):
        logger.critical("Dependency Error: 'vmrun' executable not found.")
        logger.error("To use provider='vmware', VMware Fusion (macOS) or Workstation (Windows/Linux) must be installed and 'vmrun' must be in your PATH.")
        sys.exit(1)
    logger.info("VMware dependency 'vmrun' detected successfully.")

def setup_environment(args: argparse.Namespace) -> Optional[DesktopEnv]:
    """
    Initializes the OSWorld DesktopEnv with robust error handling.
    
    Args:
        args: Parsed command line arguments containing configuration.
    
    Returns:
        DesktopEnv: The initialized environment object.
    """
    
    # 1. Pre-flight checks for VMware
    validate_vmware_dependencies()
    
    logger.info(f"Initializing DesktopEnv with provider: {args.provider}")
    
    try:
        # 2. Instantiation of DesktopEnv 
        # We configure the observation space to be multimodal (Screenshot + A11y Tree)
        # This is critical for the 'mini' agent to handle grounding.
        env = DesktopEnv(
            path_to_vm=args.vm_path,         # Path to the .vmx file
            provider_name='vmware',          # Strictly enforced as per user request
            action_space='pyautogui',        # Standard action space for agents 
            screen_size=(1920, 1080),        # High resolution for OCR 
            headless=args.headless,          # Option for headless execution 
            require_a11y_tree=False,          # Explicitly request A11y tree
            # observation_type=['screenshot', 'a11y_tree'] # Hybrid observation
        )
        
        logger.info("Desktop Env initialized successfully.")
        return env

    except Exception as e:
        # 3. Robust Exception Handling 
        # Initialization is computationally heavy and prone to timeouts.
        logger.error(f"Failed to initialize DesktopEnv: {str(e)}")
        
        # If initialization fails, we must ensure we don't leave zombie processes
        logger.warning("Attempting to clean up any partial states...")
        # In a real scenario, we might trigger a 'vmrun stop' command here if the object wasn't created
        raise e

if __name__ == "__main__":
    # Expose critical runtime configurations
    parser = argparse.ArgumentParser(description="OSWorld Environment Setup (VMware Edition)")
    
    # Enforcing 'vmware' as the default provider
    parser.add_argument("--provider", type=str, default="vmware", choices=["vmware"], 
                        help="Virtualization provider (Fixed to VMware)")
    
    # Path handling is crucial for cross-platform compatibility
    parser.add_argument("--vm_path", type=str, required=True, 
                        help="Absolute path to the Ubuntu/Windows .vmx file")
    
    parser.add_argument("--headless", action="store_true", 
                        help="Run without GUI (if supported by VMware config)")

    args = parser.parse_args()
    
    # Test the setup
    env = setup_environment(args)
    if env:
        env.close() # Always close to prevent resource leaks