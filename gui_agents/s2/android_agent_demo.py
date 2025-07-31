#!/usr/bin/env python3
"""
Demo script for AgentS2 + AndroidExecutor integration.
Shows how to run the full agent pipeline for Android UI automation.
"""

import argparse
import logging
import os
import sys
from typing import Dict

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.android_executor import AndroidExecutor
from gui_agents.s2.agents.grounding import OSWorldACI

# Import android_world components
try:
    from android_world.android_world.env.interface import AsyncAndroidEnv
    from android_world.android_world.env.android_world_controller import AndroidWorldController
    from android_world.android_world.env.config_classes import AndroidEnvConfig
except ImportError as e:
    print(f"Error importing android_world: {e}")
    print("Make sure android_world is installed and in your PYTHONPATH")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_android_env() -> AsyncAndroidEnv:
    """Create and configure the Android environment."""
    # This is a minimal configuration - you may need to adjust based on your setup
    config = AndroidEnvConfig(
        # Add your specific configuration here
        # For now, using minimal defaults
    )
    
    # Create the controller
    controller = AndroidWorldController(config)
    
    # Create the async environment
    env = AsyncAndroidEnv(controller)
    
    return env

def run_android_agent_demo(task_instruction: str = "Test turning Wi-Fi on and off"):
    """Run the full Android agent pipeline demo."""
    
    print(f"Starting Android Agent Demo with task: {task_instruction}")
    
    # Step 1: Create the Android environment
    print("Creating Android environment...")
    try:
        android_env = create_android_env()
        print("✓ Android environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create Android environment: {e}")
        return
    
    # Step 2: Create the AndroidExecutor
    print("Creating AndroidExecutor...")
    try:
        android_executor = AndroidExecutor(android_env)
        print("✓ AndroidExecutor created successfully")
    except Exception as e:
        print(f"✗ Failed to create AndroidExecutor: {e}")
        return
    
    # Step 3: Set up engine parameters (you'll need to configure these)
    engine_params = {
        "engine_type": "anthropic",  # or "openai"
        "model": "claude-3-5-sonnet-20241022",  # or your preferred model
        "api_key": os.getenv("ANTHROPIC_API_KEY"),  # Set this environment variable
    }
    
    # Step 4: Create a dummy grounding agent (not used for Android but required by AgentS2)
    # In a real implementation, you might want to create a proper Android grounding agent
    dummy_grounding_agent = None  # This will need to be properly configured
    
    # Step 5: Create AgentS2 with AndroidExecutor
    print("Creating AgentS2 with AndroidExecutor...")
    try:
        agent = AgentS2(
            engine_params=engine_params,
            grounding_agent=dummy_grounding_agent,
            platform="android",  # Set platform to android
            action_space="android",  # Set action space to android
            observation_type="mixed",
            search_engine=None,
            android_executor=android_executor,  # Use our AndroidExecutor
        )
        print("✓ AgentS2 created successfully with AndroidExecutor")
    except Exception as e:
        print(f"✗ Failed to create AgentS2: {e}")
        return
    
    # Step 6: Run the agent pipeline
    print(f"Running agent pipeline for task: {task_instruction}")
    
    # Create a dummy observation (in real usage, this would come from the Android environment)
    observation = {
        "screenshot": None,  # Would be actual screenshot from Android
        "ui_tree": None,     # Would be actual UI tree from Android
    }
    
    try:
        # Reset the agent
        agent.reset()
        
        # Run the prediction loop
        max_steps = 10
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Get next action from agent
            info, actions = agent.predict(
                instruction=task_instruction,
                observation=observation
            )
            
            print(f"Agent info: {info}")
            print(f"Actions: {actions}")
            
            # Check if task is complete
            if any("done" in str(action).lower() for action in actions):
                print("✓ Task completed!")
                break
            elif any("fail" in str(action).lower() for action in actions):
                print("✗ Task failed!")
                break
                
    except Exception as e:
        print(f"✗ Error during agent execution: {e}")
    
    # Step 7: Cleanup
    print("Cleaning up...")
    try:
        android_env.close()
        print("✓ Android environment closed successfully")
    except Exception as e:
        print(f"✗ Error closing Android environment: {e}")

def main():
    parser = argparse.ArgumentParser(description="Android Agent Demo")
    parser.add_argument(
        "--task",
        type=str,
        default="Test turning Wi-Fi on and off",
        help="Task instruction for the agent"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY environment variable not set")
        print("You may need to set this for the agent to work properly")
    
    run_android_agent_demo(args.task)

if __name__ == "__main__":
    main() 