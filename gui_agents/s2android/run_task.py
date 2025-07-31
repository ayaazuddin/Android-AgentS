#!/usr/bin/env python3
"""
Run Android contact tasks using Android Agent-S2.

This script demonstrates the Android Agent-S2 system on contact-related tasks
using the actual Android emulator. Defaults to adding a new contact.
"""

import os
import sys
import logging
from typing import Type

# Add the project root and android_world to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
android_world_path = os.path.join(project_root, 'android_world')
sys.path.insert(0, project_root)
sys.path.insert(0, android_world_path)

from absl import app
from absl import flags
from absl import logging as absl_logging

# Try to import android_world components
try:
    from android_world import registry
    from android_world.env import env_launcher
    from android_world.task_evals import task_eval
    print("✅ Successfully imported android_world components")
except ImportError as e:
    print(f"❌ Error importing android_world: {e}")
    print("Please ensure android_world is properly installed or in the correct path")
    sys.exit(1)

# Import our Android Agent-S2
from gui_agents.s2android.agents.agent_s import AndroidAgentS2

# Configure logging
logging.basicConfig(level=logging.INFO)
absl_logging.set_verbosity(absl_logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('C:/Users/owner/AppData/Local/Android/Sdk/platform-tools/adb.exe'),
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found in the common Android SDK paths. Please install Android'
        " SDK and ensure adb is in one of the expected directories. If it's"
        ' already installed, point to the installed location.'
    )


# Define command line flags
_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model',
    'gpt-4o-mini',
    'The LLM model to use for the agent.',
)
_API_KEY = flags.DEFINE_string(
    'api_key',
    None,
    'OpenAI API key. If not provided, will try to get from environment.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run. If not provided, a random task will be selected.',
)


def _main() -> None:
    """Run Android contact tasks using Android Agent-S2."""
    
    # Get API key
    api_key = _API_KEY.value
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")
    
    print("🚀 Starting Android Agent-S2 Contact Task")
    print(f"📱 Using device on port: {_DEVICE_CONSOLE_PORT.value}")
    print(f"🤖 Using model: {_MODEL_NAME.value}")
    
    # Load and setup the Android environment
    print("🔧 Setting up Android environment...")
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    env.reset(go_home=True)
    
    # Get the task registry and select a random task
    print("📋 Loading task registry...")
    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    
    # Select task type (random or specific)
    import random
    if _TASK.value:
        if _TASK.value not in aw_registry:
            raise ValueError(f'Task {_TASK.value} not found in registry.')
        task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
        print(f"🎯 Using specified task: {_TASK.value}")
    else:
        # Default to ContactsAddContact task
        task_type: Type[task_eval.TaskEval] = aw_registry["ContactsAddContact"]
        print(f"🎯 Using default task: ContactsAddContact")
    params = task_type.generate_random_params()
    task = task_type(params)
    task.initialize_task(env)
    
    print(f"🎯 Task Goal: {task.goal}")
    
    # Create the Android Agent-S2
    print("🤖 Creating Android Agent-S2...")
    engine_params = {
        "engine_type": "openai",
        "model": _MODEL_NAME.value,
        "api_key": api_key,
        "temperature": 0.1,
        "max_tokens": 1000,
    }




    
    agent = AndroidAgentS2(
        engine_params=engine_params,
        action_space="android",
        observation_type="mixed",
        search_engine=None,
        android_env=env,
        use_default_kb=False,
        memory_root_path=os.path.join(os.path.dirname(__file__), "memory"),
        embedding_engine_type="openai",
        embedding_engine_params={"api_key": api_key},
    )
    
    # Reset the agent
    agent.reset()
    
    print("🔄 Starting task execution...")
    print("=" * 50)
    print(f"🎯 Goal: {task.goal}")
    print("=" * 50)
    
    # Execute the task
    is_done = False
    max_steps = int(task.complexity * 10)  # Allow more steps for complex reasoning
    
    # Initialize episode data collection
    episode_frames = []
    episode_actions = []
    
    for step in range(max_steps):
        print(f"\n📝 Step {step + 1}/{max_steps}")
        print("-" * 30)
        
        try:
            # Get the current state
            state = env.get_state(wait_to_stabilize=False)
            
            # Create observation dictionary with device context
            observation = {
                'ui_elements': state.ui_elements,
                'pixels': state.pixels,
                'screenshot': None,  # Skip screenshot for now to avoid ndarray issues
                'current_activity': env.foreground_activity_name,
                'screen_size': env.device_screen_size,
                'device_info': {
                    'screen_size': env.device_screen_size,
                    'current_activity': env.foreground_activity_name,
                    'num_ui_elements': len(state.ui_elements) if state.ui_elements else 0,
                    'device_model': 'Android Emulator',  # Could be made dynamic
                    'android_version': 'API 34',  # Could be made dynamic
                }
            }
            
            # Collect frame data for episode analysis
            frame_data = {
                'ui_elements': state.ui_elements,
                'screenshot': None,  # Will be added if screenshot is available
                'device_info': observation['device_info']
            }
            episode_frames.append(frame_data)
            
            # Execute one step using predict method
            agent_info, actions = agent.predict(task.goal, observation)
            
            # Execute the actions if they exist and are not DONE
            if actions and actions != ["DONE"]:
                print(f"🔄 Executing actions: {actions}")
                # Here you would execute the actions on the environment
                # For now, we'll just log them
                for action in actions:
                    print(f"   📤 Action: {action}")
                    # Collect action data for episode analysis
                    action_data = {
                        'action_type': action.get('action_type', 'unknown'),
                        'index': action.get('index', 'N/A'),
                        'text': action.get('text', ''),
                        'reason': action.get('reason', ''),
                        'step': step + 1
                    }
                    episode_actions.append(action_data)
            
            # Get updated state after action execution
            updated_state = env.get_state(wait_to_stabilize=False)
            updated_observation = {
                'ui_elements': updated_state.ui_elements,
                'pixels': updated_state.pixels,
                'screenshot': None,
                'current_activity': env.foreground_activity_name,
                'screen_size': env.device_screen_size,
                'device_info': {
                    'screen_size': env.device_screen_size,
                    'current_activity': env.foreground_activity_name,
                    'num_ui_elements': len(updated_state.ui_elements) if updated_state.ui_elements else 0,
                    'device_model': 'Android Emulator',
                    'android_version': 'API 34',
                }
            }
            
            # Verify the action execution using the updated state
            if agent_info.get('current_subtask'):
                manager_goal = f"{agent_info.get('current_subtask')}: {agent_info.get('current_subtask_info', '')}"
                is_success, reasoning, verification_info = agent.verifier.verify_execution(
                    manager_goal=manager_goal,
                    worker_execution_result=agent_info,
                    current_ui_state=updated_observation
                )
                
                print(f"Verifier rates the result of the action")
            
            # Check if the main goal has been achieved using the task's success condition
            try:
                if task.is_successful(env):
                    print("🎯 Main goal achieved! Task completed successfully.")
                    actions = ["DONE"]
                    is_done = True
            except Exception as e:
                # If we can't check success, continue normally
                pass
            
            # Extract current subtask info
            current_subtask = agent_info.get('current_subtask', 'Unknown')
            current_subtask_info = agent_info.get('current_subtask_info', '')
            
            # Show actions in a clean way
            if actions and actions != ["DONE"]:
                for action in actions:
                    if "SUCCESS" in action:
                        print(f"✅ {action}")
                    elif "FAIL" in action:
                        print(f"❌ {action}")
                    else:
                        print(f"🔄 {action}")
            elif "DONE" in actions:
                print("✅ Task completed!")
                is_done = True
                break
                
        except Exception as e:
            print(f"❌ Error in step {step + 1}: {e}")
            break
    
    # Check if the task was successful
    print("\n" + "=" * 50)
    print("📊 Task Results")
    print("=" * 50)
    
    try:
        success_score = task.is_successful(env)
        agent_successful = is_done and success_score == 1
        
        print(f"🎯 Goal: {task.goal}")
        print(f"📈 Success Score: {success_score}/1.0")
        print(f"✅ Agent Completed: {'Yes' if is_done else 'No'}")
        print(f"🏆 Result: {'SUCCESS ✅' if agent_successful else 'FAILED ❌'}")
        
    except Exception as e:
        print(f"❌ Error evaluating task: {e}")
        agent_successful = False
    
    # Clean up
    print("\n🧹 Cleaning up...")
    env.close()
    
    # Initialize supervisor agent for episode review
    try:
        from gui_agents.s2android.agents.supervisor_agent import AndroidSupervisorAgent
        
        print("\n🔍 Initializing Supervisor Agent for episode review...")
        supervisor = AndroidSupervisorAgent(engine_params)
        
        # Review the episode with collected data
        print("📝 Conducting episode review...")
        print(f"   📊 Collected {len(episode_frames)} frames and {len(episode_actions)} actions")
        review_summary, review_details = supervisor.review_episode(
            task_description=task.goal,
            episode_frames=episode_frames,
            episode_actions=episode_actions,
            final_success=agent_successful
        )
        
        # Print the review
        supervisor.print_review(review_summary, review_details)
        
    except Exception as e:
        print(f"⚠️  Supervisor agent review failed: {e}")
    
    print("🎉 Done!")
    return agent_successful


def main(argv):
    """Main entry point."""
    del argv
    return _main()


if __name__ == '__main__':
    app.run(main) 