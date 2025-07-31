#!/usr/bin/env python3
"""
Demo script for AndroidAgentS - Agent-S with android_world compatibility.
This demonstrates using Agent-S's modular architecture with android_world's expected interface.
"""

import argparse
import logging
import sys
import os

# Add android_world to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
android_world_path = os.path.join(project_root, 'android_world')
sys.path.insert(0, android_world_path)

from android_world.env import env_launcher
from android_world.task_evals.single.system import SystemWifiTurnOn
from android_world.agents import infer
from absl import flags
from agents.android_agent_s import AndroidAgentS

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

def create_android_env(console_port: int = 5554, perform_emulator_setup: bool = False):
    """Create and setup the Android environment."""
    print("Setting up Android environment...")
    
    # adb_dir = 
    # if not os.path.exists(adb_dir):
    #     print(f"Warning: ADB directory not found at {adb_dir}")
    #     adb_dir = None
    
    adb_path = find_adb_directory()
    print(f"Using ADB at: {adb_path}")
    # Load and setup the environment
    env = env_launcher.load_and_setup_env(
        console_port=console_port,
        emulator_setup=perform_emulator_setup,
        adb_path=adb_path,
    )
    
    print("✓ Android environment created successfully!")
    return env

def create_wifi_task():
    """Create the Wi-Fi turn on task."""
    print("Creating Wi-Fi task...")
    # Generate the required parameters for the task
    params = SystemWifiTurnOn.generate_random_params()
    task = SystemWifiTurnOn(params)
    print("✓ Wi-Fi task created successfully!")
    return task

def run_android_agent_s_demo(console_port: int = 5554, perform_emulator_setup: bool = False, api_key: str = None):
    """Run the AndroidAgentS demo with the Wi-Fi task."""
    
    # Get API key from environment variable if not provided
    
    
    if not api_key:
        print("Error: API key is required for LLM access. Set OPENAI_API_KEY environment variable or use --api_key")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create environment
        env = create_android_env(console_port, perform_emulator_setup)
        
        # Create task
        task = create_wifi_task()
        
        # Initialize task
        task.initialize_task(env)
        
        # Create LLM wrapper
        engine_params = {
            "engine_type": "openai",
            "model": "gpt-4o",  # Using OpenAI's GPT-4o model
            "api_key": api_key,
            "temperature": 0.1,
            "max_tokens": 1000,
        }
        
        # Create AndroidAgentS
        print("Creating AndroidAgentS...")
        agent = AndroidAgentS(
            env=env,
            engine_params=engine_params,
            name="AndroidAgentS-WiFi",
        )
        
        # Reset agent
        agent.reset(go_home=True)
        
        # Get task goal
        goal = task.template
        print(f"Task goal: {goal}")
        
        # Run the agent
        print("\n" + "="*50)
        print("Starting AndroidAgentS execution...")
        print("="*50)
        
        max_steps = 20
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            
            # Execute one step
            result = agent.step(goal)
            
            # Print step info
            print(f"Action: {result.data.get('action_output', 'None')}")
            print(f"Summary: {result.data.get('summary', 'None')}")
            
            # Check if done
            if result.done:
                print("✓ Agent reports task completed!")
                break
                
            # Check for errors
            if 'error' in result.data:
                print(f"✗ Error: {result.data['error']}")
                break
        
        # Verify task completion
        print("\n" + "="*50)
        print("Verifying task completion...")
        print("="*50)
        
        if task.is_successful(env):
            print("✓ Task verification: SUCCESS!")
            print("Wi-Fi has been successfully turned on!")
        else:
            print("✗ Task verification: FAILED")
            print("Wi-Fi was not turned on successfully.")
            
    except Exception as e:
        print(f"Error during demo execution: {e}")
        import traceback
        traceback.print_exc()

def main():
    # parser = argparse.ArgumentParser(description="AndroidAgentS Demo")
    # parser.add_argument("--console_port", type=int, default=5554, 
    #                    help="Android emulator console port")
    # parser.add_argument("--perform_emulator_setup", action="store_true",
    #                    help="Perform emulator setup")
    # parser.add_argument("--api_key", type=str, required=True,
    #                    help="OpenAI API key for LLM access")
    
    
    print("AndroidAgentS Demo")
    print("="*50)
    print("This demo uses Agent-S's modular architecture with android_world compatibility")
    print("="*50)

    api_key = os.environ.get('OPENAI_API_KEY')
    
    run_android_agent_s_demo(api_key=api_key)

if __name__ == "__main__":
    main() 