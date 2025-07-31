#!/usr/bin/env python3
"""
Real Android Agent Demo using AgentS2 + AndroidExecutor with actual android_world.
Runs a real Wi-Fi task using the SystemWifiTurnOn task from android_world.
"""

import argparse
import logging
import os
import sys
from typing import Dict

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.agent_s import AgentS2
from agents.android_executor import AndroidExecutor

# Import android_world components
try:
    from android_world.env import env_launcher
    from android_world.task_evals.single.system import SystemWifiTurnOn
    from android_world.task_evals import task_eval
except ImportError as e:
    print(f"Error importing android_world: {e}")
    print("Make sure android_world is installed and in your PYTHONPATH")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_adb_directory() -> str:
    """Find the ADB executable path."""
    potential_paths = [
        os.path.expanduser('C:/Users/owner/AppData/Local/Android/Sdk/platform-tools/adb.exe'),
        '/usr/local/bin/adb',
        '/usr/bin/adb',
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found. Please install Android SDK and ensure adb is available.'
    )

def create_android_env(console_port: int = 5554, perform_emulator_setup: bool = False) -> any:
    """Create and configure the Android environment."""
    try:
        adb_path = find_adb_directory()
        print(f"Using ADB at: {adb_path}")
        
        # Set environment variables
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['GRPC_TRACE'] = 'none'
        
        # Load and setup environment
        env = env_launcher.load_and_setup_env(
            console_port=console_port,
            emulator_setup=perform_emulator_setup,
            adb_path=adb_path,
        )
        
        # Reset environment
        env.reset(go_home=True)
        print("✓ Android environment created and reset successfully")
        return env
        
    except Exception as e:
        print(f"✗ Failed to create Android environment: {e}")
        raise

def create_wifi_task() -> task_eval.TaskEval:
    """Create a Wi-Fi turn on task."""
    task_type = SystemWifiTurnOn
    params = task_type.generate_random_params()
    task = task_type(params)
    print(f"✓ Created task: {task.goal}")
    return task

def run_android_agent_demo(
    console_port: int = 5554,
    perform_emulator_setup: bool = False,
    api_key: str = None
):
    """Run the full Android agent pipeline demo with real environment."""
    
    print("=== Real Android Agent Demo ===")
    print(f"Console port: {console_port}")
    print(f"Emulator setup: {perform_emulator_setup}")
    
    # Step 1: Create the Android environment
    print("\n1. Creating Android environment...")
    try:
        android_env = create_android_env(console_port, perform_emulator_setup)
    except Exception as e:
        print(f"✗ Failed to create Android environment: {e}")
        print("Make sure you have:")
        print("1. Android SDK installed with ADB")
        print("2. Android emulator running")
        print("3. Correct console port (default: 5554)")
        return
    
    # Step 2: Create the Wi-Fi task
    print("\n2. Creating Wi-Fi task...")
    try:
        task = create_wifi_task()
        task.initialize_task(android_env)
        print(f"✓ Task initialized: {task.goal}")
    except Exception as e:
        print(f"✗ Failed to create task: {e}")
        android_env.close()
        return
    
    # Step 3: Create the AndroidExecutor
    print("\n3. Creating AndroidExecutor...")
    try:
        android_executor = AndroidExecutor(android_env)
        print("✓ AndroidExecutor created successfully")
    except Exception as e:
        print(f"✗ Failed to create AndroidExecutor: {e}")
        android_env.close()
        return
    
    # Step 4: Set up engine parameters
    print("\n4. Setting up engine parameters...")
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("✗ No API key provided. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
            android_env.close()
            return
    
    engine_params = {
        "engine_type": "openai",  # or "openai"
        "model": "claude-3-5-sonnet-20241022",
        "api_key": api_key,
    }
    print("✓ Engine parameters configured")
    
    # Step 5: Create a dummy grounding agent (required by AgentS2 but not used for Android)
    print("\n5. Creating AgentS2 with AndroidExecutor...")
    try:
        # Create a minimal mock grounding agent
        class MockGroundingAgent:
            def __init__(self):
                pass
            def get_active_apps(self, obs):
                return ["settings"]
            def get_top_app(self, obs):
                return "settings"
            def linearize_and_annotate_tree(self, obs):
                return "Mock UI tree"
        
        mock_grounding_agent = MockGroundingAgent()
        
        agent = AgentS2(
            engine_params=engine_params,
            grounding_agent=mock_grounding_agent,
            platform="android",
            action_space="android",
            observation_type="mixed",
            search_engine=None,
            android_executor=android_executor,
        )
        print("✓ AgentS2 created successfully with AndroidExecutor")
    except Exception as e:
        print(f"✗ Failed to create AgentS2: {e}")
        android_env.close()
        return
    
    # Step 6: Run the agent pipeline
    print(f"\n6. Running agent pipeline for task: {task.goal}")
    
    try:
        # Reset the agent
        agent.reset()
        
        # Run the prediction loop
        max_steps = 15
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Get current state from Android environment
            state = android_env.get_state(wait_to_stabilize=True)
            
            # Create observation for agent
            observation = {
                "screenshot": state.pixels if hasattr(state, 'pixels') else None,
                "ui_tree": state.ui_elements if hasattr(state, 'ui_elements') else [],
            }
            
            # Get next action from agent
            info, actions = agent.predict(
                instruction=task.goal,
                observation=observation
            )
            
            print(f"Agent info: {info}")
            print(f"Actions: {actions}")
            
            # Check if task is complete
            if any("done" in str(action).lower() for action in actions):
                print("✓ Agent reports task completed!")
                break
            elif any("fail" in str(action).lower() for action in actions):
                print("✗ Agent reports task failed!")
                break
            elif any("wait" in str(action).lower() for action in actions):
                print("⏳ Agent is waiting...")
                import time
                time.sleep(2)  # Wait 2 seconds before next step
                continue
                
        # Check if task was actually successful
        print("\n7. Verifying task completion...")
        try:
            success_score = task.is_successful(android_env)
            print(f"Task success score: {success_score}")
            if success_score == 1.0:
                print("✅ Task completed successfully!")
            else:
                print("❌ Task failed!")
        except Exception as e:
            print(f"✗ Error checking task success: {e}")
                
    except Exception as e:
        print(f"✗ Error during agent execution: {e}")
    
    # Step 7: Cleanup
    print("\n8. Cleaning up...")
    try:
        android_env.close()
        print("✓ Android environment closed successfully")
    except Exception as e:
        print(f"✗ Error closing Android environment: {e}")

def main():
    parser = argparse.ArgumentParser(description="Real Android Agent Demo")
    parser.add_argument(
        "--console_port",
        type=int,
        default=5554,
        help="Console port of the Android emulator (default: 5554)"
    )
    parser.add_argument(
        "--perform_emulator_setup",
        action="store_true",
        help="Perform emulator setup (only needed once)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="API key for the LLM (or set ANTHROPIC_API_KEY/OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    run_android_agent_demo(
        console_port=args.console_port,
        perform_emulator_setup=args.perform_emulator_setup,
        api_key=args.api_key
    )

if __name__ == "__main__":
    main() 