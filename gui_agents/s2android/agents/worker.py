import logging
import re
import textwrap
from typing import Dict, List, Tuple, Optional, Any

from gui_agents.s2.core.module import BaseModule
from gui_agents.s2android.memory.procedural_memory_android import PROCEDURAL_MEMORY_ANDROID as PROCEDURAL_MEMORY
from gui_agents.s2.utils.common_utils import (
    Node,
    calculate_tokens,
    call_llm_safe,
    parse_single_code_from_string,
    sanitize_code,
    extract_first_agent_function,
)

logger = logging.getLogger("androidenv.worker")


class AndroidWorker(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent,
        local_kb_path: str,
        embedding_engine,
        platform: str = "android",
        enable_reflection: bool = True,
        use_subtask_experience: bool = True,
    ):
        """
        Android Worker receives a subtask and generates the next action by finding UI elements.
        Args:
            engine_params: Dict
                Parameters for the multimodal engine
            grounding_agent: AndroidACI
                The Android grounding agent to use
            local_kb_path: str
                Path to knowledge base
            platform: str
                Platform (always "android" for this worker)
            enable_reflection: bool
                Whether to enable reflection
            use_subtask_experience: bool
                Whether to use subtask experience
        """
        super().__init__(engine_params, platform)

        self.grounding_agent = grounding_agent
        self.local_kb_path = local_kb_path
        self.embedding_engine = embedding_engine
        self.enable_reflection = enable_reflection
        self.use_subtask_experience = use_subtask_experience
        self.reset()

    def reset(self):
        # Android-specific actions (no desktop-specific actions like set_cell_values)
        skipped_actions = []

        sys_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
            type(self.grounding_agent), skipped_actions=skipped_actions
        ).replace("CURRENT_OS", "android")

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        # Create element finding agent for UI element grounding
        self.element_finding_agent = self._create_agent(
            self._get_element_finding_prompt()
        )

        # Disable knowledge base for Android to avoid file dependencies
        self.knowledge_base = None

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []
        self.planner_history = []
        self.max_trajector_length = 8

    def _get_element_finding_prompt(self) -> str:
        """Get the system prompt for element finding"""
        return """You are an Android UI element finder. Your job is to find the most relevant UI element from a list based on a natural language description.

Given a description and a list of UI elements, return ONLY the element index (0-based) that best matches the description.

Rules:
1. Return only a single number (the element index)
2. Consider element text, type, and interactive properties
3. Prefer clickable elements for click actions
4. Prefer editable elements for type actions
5. If no good match is found, return -1

Example:
Description: "click the settings button"
Elements: 
0: Settings (Button) [clickable]
1: Home (Button) [clickable]
2: Back (Button) [clickable]

Response: 0"""

    def find_element_by_description(self, description: str, obs: Dict) -> Optional[int]:
        """Find UI element by natural language description using LLM"""
        ui_elements = obs.get("ui_elements", [])
        if not ui_elements:
            logger.warning("No UI elements available for grounding")
            return None
            
        # Create a text representation of UI elements
        ui_text = self._linearize_ui_elements(ui_elements)
        
        # Use LLM to find the best matching element
        prompt = f"""Description: "{description}"

Find the most relevant UI element from this list:
{ui_text}

Return only the element index (0-based) that best matches the description."""
        
        self.element_finding_agent.reset()
        self.element_finding_agent.add_message(text_content=prompt)
        
        if obs.get("screenshot") and obs["screenshot"] is not None:
            self.element_finding_agent.add_message(
                text_content="Screenshot:", 
                image_content=obs["screenshot"], 
                put_text_last=True
            )
        
        response = call_llm_safe(self.element_finding_agent)
        logger.info(f"Element finding response: {response}")
        
        # Extract element index from response
        try:
            element_idx = int(re.findall(r'-?\d+', response)[0])
            if element_idx == -1:
                logger.warning("No suitable element found for description")
                return None
            if 0 <= element_idx < len(ui_elements):
                return element_idx
            else:
                logger.warning(f"Element index {element_idx} out of range (0-{len(ui_elements)-1})")
                return None
        except (ValueError, IndexError):
            logger.warning(f"Could not parse element index from response: {response}")
            return None

    def _linearize_ui_elements(self, ui_elements: List[Any]) -> str:
        """Convert UI elements to a readable text format"""
        if not ui_elements:
            return "No UI elements available"
            
        ui_text = "UI Elements:\n"
        for idx, element in enumerate(ui_elements):
            text = getattr(element, 'text', 'No text')
            element_type = getattr(element, 'class_name', 'Unknown')
            hint_text = getattr(element, 'hint_text', '')
            content_description = getattr(element, 'content_description', '')
            clickable = getattr(element, 'is_clickable', False)
            editable = getattr(element, 'is_editable', False)
            focusable = getattr(element, 'is_focusable', False)
            checkable = getattr(element, 'is_checkable', False)
            resource_id = getattr(element, 'resource_id', '')
            
            ui_text += f"{idx}: {text} ({element_type})"
            if hint_text:
                ui_text += f" [hint: '{hint_text}']"
            if content_description:
                ui_text += f" [desc: '{content_description}']"
            if resource_id:
                ui_text += f" [id: '{resource_id}']"
            if clickable:
                ui_text += " [clickable]"
            if editable:
                ui_text += " [editable]"
            if focusable:
                ui_text += " [focusable]"
            if checkable:
                ui_text += " [checkable]"
            ui_text += "\n"
            
        return ui_text



    def flush_messages(self):
        # generator msgs are alternating [user, assistant], so 2 per round
        if len(self.generator_agent.messages) > 2 * self.max_trajector_length + 1:
            self.generator_agent.remove_message_at(1)
            self.generator_agent.remove_message_at(1)
        # reflector msgs are all [(user text, user image)], so 1 per round
        if len(self.reflection_agent.messages) > self.max_trajector_length + 1:
            self.reflection_agent.remove_message_at(1)

    def generate_next_action(
        self,
        instruction: str,
        search_query: str,
        subtask: str,
        subtask_info: Dict,
        future_tasks: List[Node],
        done_task: List[Node],
        obs: Dict,
    ) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """
        
        # Always show worker reasoning
        if self.turn_count == 0:
            print(f"Worker → {subtask}")
        
        # Store previous state for change detection
        previous_activity = getattr(self, '_previous_activity', None)
        previous_ui_elements = getattr(self, '_previous_ui_elements', [])
        
        # Get current state
        current_activity = obs.get('current_activity', 'Unknown')
        current_ui_elements = obs.get('ui_elements', [])
        
        # Detect screen changes
        screen_changed = False
        if previous_activity and previous_activity != current_activity:
            screen_changed = True
        elif len(current_ui_elements) != len(previous_ui_elements):
            screen_changed = True
        elif self.turn_count > 0:
            # Also check for significant UI element changes (not just count)
            if len(previous_ui_elements) > 0 and len(current_ui_elements) > 0:
                # Check if the first few UI elements have changed (indicating navigation)
                prev_texts = [getattr(elem, 'text', '') for elem in previous_ui_elements[:3]]
                curr_texts = [getattr(elem, 'text', '') for elem in current_ui_elements[:3]]
                if prev_texts != curr_texts:
                    screen_changed = True
        
        # Store current state for next comparison
        self._previous_activity = current_activity
        self._previous_ui_elements = current_ui_elements
        
        # Add state stability check like android_world T3A
        if self.turn_count > 0 and not screen_changed:
            # Wait for UI to stabilize before proceeding
            import time
            time.sleep(2.0)  # Wait longer for UI to stabilize (increased from 0.5s)
            
            # Check if state changed after waiting
            new_activity = obs.get('current_activity', 'Unknown')
            new_ui_elements = obs.get('ui_elements', [])
            
            if new_activity != current_activity or len(new_ui_elements) != len(current_ui_elements):
                screen_changed = True
                # Update current state with stabilized state
                current_activity = new_activity
                current_ui_elements = new_ui_elements
                self._previous_activity = current_activity
                self._previous_ui_elements = current_ui_elements

        # Get RAG knowledge, only update system message at t=0
        if self.turn_count == 0:
            if self.use_subtask_experience:
                subtask_query_key = (
                    "Task:\n"
                    + search_query
                    + "\n\nSubtask: "
                    + subtask
                    + "\nSubtask Instruction: "
                    + subtask_info
                )
                # Skip knowledge base retrieval for Android
                retrieved_similar_subtask = "None"
                retrieved_subtask_experience = "None"
                logger.info("SKIPPING KNOWLEDGE BASE RETRIEVAL FOR ANDROID")

            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "SUBTASK_DESCRIPTION", subtask
                )
                .replace("TASK_DESCRIPTION", instruction)
                .replace("FUTURE_TASKS", ", ".join([f.name for f in future_tasks]))
                .replace("DONE_TASKS", ",".join(d.name for d in done_task))
            )

        # Reflection generation
        reflection = None
        if self.enable_reflection:
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Subtask Description: {subtask}
                    Subtask Information: {subtask_info}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial screen is provided. No action has been taken yet.",
                    image_content=obs.get("screenshot"),
                    role="user",
                )
            else:
                text_content = self.clean_worker_generation_for_reflection(
                    self.planner_history[-1]
                )
                self.reflection_agent.add_message(
                    text_content=text_content,
                    image_content=obs.get("screenshot"),
                    role="user",
                )
                reflection = call_llm_safe(self.reflection_agent)
                self.reflections.append(reflection)
                logger.info("REFLECTION: %s", reflection)

        # Add device context to the message
        device_info = obs.get('device_info', {})
        
        # Debug: Show what's in the observation
        print(f"   🔍 Observation keys: {list(obs.keys())}")
        print(f"   🔍 Device info: {device_info}")
        
        # Get UI element summaries for better context
        ui_elements = obs.get('ui_elements', [])
        ui_summary = ""
        if ui_elements:
            # Show more detailed UI element information
            element_descriptions = []
            for i, element in enumerate(ui_elements[:10]):  # Show first 10 elements
                element_text = getattr(element, 'text', '')
                element_desc = getattr(element, 'content_description', '')
                element_class = getattr(element, 'class_name', '')
                element_hint = getattr(element, 'hint_text', '')
                element_editable = getattr(element, 'is_editable', False)
                
                # Create a more detailed description with field identification
                description = f"Element {i}: '{element_text}'"
                if element_hint:
                    description += f" [hint: '{element_hint}']"
                if element_desc:
                    description += f" [desc: '{element_desc}']"
                if element_editable:
                    description += " [editable]"
                description += f" ({element_class})"
                
                element_descriptions.append(description)
                
                # Debug: Show raw element data for first few elements
                if i < 3:
                    print(f"   🔍 Element {i} raw data: text='{element_text}', hint='{element_hint}', desc='{element_desc}', class='{element_class}', editable={element_editable}")
            
            ui_summary = "\nAvailable UI Elements:\n" + "\n".join(element_descriptions)
            if len(ui_elements) > 10:
                ui_summary += f"\n... and {len(ui_elements) - 10} more elements"
                
            # Add helpful hints about what to look for
            ui_summary += "\n\nUI Element Hints:"
            ui_summary += "\n- Look for 'Settings', 'System Preferences', 'Gear icon' for settings"
            ui_summary += "\n- Look for 'WiFi', 'Internet', 'Network', 'Wireless' for WiFi settings"
            ui_summary += "\n- Look for 'Quick Settings', 'Control Center', 'Settings Panel' for quick access"
            ui_summary += "\n- Look for 'App Drawer', 'All Apps', 'Applications' for app list"
            ui_summary += "\n- Look for 'Notifications', 'Alerts', 'Bell icon' for notification panel"
            ui_summary += "\n- Look for 'Home', 'Back', 'Close', 'Done' for navigation"
            ui_summary += "\n- Look for 'Search', 'Find', 'Magnifying glass' for search functionality"
            ui_summary += "\n- Look for 'Menu', 'More', 'Options', 'Three dots' for additional options"
            ui_summary += "\n\nField Identification Hints:"
            ui_summary += "\n- Input fields have [hint: '...'] showing what to enter"
            ui_summary += "\n- Phone number fields have [hint: 'Phone' or 'Mobile']"
            ui_summary += "\n- Name fields have [hint: 'First name' or 'Last name']"
            ui_summary += "\n- Company fields have [hint: 'Company']"
            ui_summary += "\n- Dropdowns (Spinner class) show options but don't accept text input"
            ui_summary += "\n- Editable fields are marked with [editable]"
        else:
            ui_summary = "\n⚠️  No UI elements available - this may indicate an issue with UI detection"

        device_context = f"""
Device Context:
- Current Activity: {device_info.get('current_activity', 'Unknown')}
- Device: {device_info.get('device_model', 'Android Device')}{ui_summary}

Available Actions (use JSON format):
- Click on a UI element: {{"action_type": "click", "index": <element_index>}}
- Long press on a UI element: {{"action_type": "long_press", "index": <element_index>}}
- Type text into a field: {{"action_type": "input_text", "text": "<text>", "index": <element_index>}}
- Swipe in a direction: {{"action_type": "swipe", "direction": "<up/down/left/right>"}}
- Scroll in a direction: {{"action_type": "scroll", "direction": "<up/down/left/right>"}}
- Open an app: {{"action_type": "open_app", "app_name": "<app_name>"}}
- Wait for screen update: {{"action_type": "wait"}}
- Navigate home: {{"action_type": "navigate_home"}}
- Navigate back: {{"action_type": "navigate_back"}}
- Complete task: {{"action_type": "status", "goal_status": "complete"}}
- Task infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
- Answer user: {{"action_type": "answer", "text": "<answer>"}}

Action Guidelines:
- Use {{"action_type": "click", "index": <index>}} for tapping UI elements
- Use {{"action_type": "input_text", "text": "<text>", "index": <index>}} for text input
- Use {{"action_type": "scroll", "direction": "<direction>"}} for scrolling (up, down, left, right)
- Use {{"action_type": "swipe", "direction": "<direction>"}} for swiping gestures
- Use {{"action_type": "open_app", "app_name": "<app_name>"}} to directly open apps like Settings, WiFi, etc.
- Use {{"action_type": "long_press", "index": <index>}} for long press actions
- Use {{"action_type": "status", "goal_status": "complete"}} when task is completed
- Use {{"action_type": "status", "goal_status": "infeasible"}} if task cannot be completed
- Be specific about element indices (0, 1, 2, etc.)
- Try easiest approach first, switch strategies if needed
- IMPORTANT: If the same action fails multiple times, try a different approach
- IMPORTANT: Look for UI elements with similar names (e.g., "Internet" = "WiFi", "Settings" = "System Preferences")
- IMPORTANT: Use open_app for app launching instead of navigating through app drawer
- **FIELD IDENTIFICATION**: Look for `hint_text` to identify input fields (e.g., "Phone" for phone numbers, "Name" for names)
- **DROPDOWN HANDLING**: Dropdowns show options like "Home/Mobile/Work" - click to select, don't type text
- **FIELD CONFUSION**: Ensure you're entering data in the correct field type (phone number → phone field, not company field)
- **NAVIGATION**: If stuck, try {{"action_type": "navigate_back"}} to return to previous screen
"""

        generator_message = (
            f"\nYou may use this reflection on the previous action and overall trajectory: {reflection}\n"
            if reflection and self.turn_count > 0
            else ""
        ) + f"Text Buffer = [{','.join(self.grounding_agent.notes)}]."

        # Add screen change information to help with reasoning
        if screen_changed:
            generator_message += f"\nSCREEN CHANGED: The screen has changed since the last action."
        elif self.turn_count > 0:
            generator_message += f"\nNO SCREEN CHANGE: The screen has not changed since the last action. Consider trying a different approach."
            
        # Add failure tracking to prevent loops
        if self.turn_count > 0:
            recent_actions = self.planner_history[-3:] if len(self.planner_history) >= 3 else self.planner_history
            repeated_actions = len(set(recent_actions)) < len(recent_actions)
            if repeated_actions:
                generator_message += f"\nWARNING: Similar actions have been repeated. Try a completely different approach."
                generator_message += f"\nSUGGESTIONS:"
                generator_message += f"\n- If swiping doesn't work, try clicking on visible elements"
                generator_message += f"\n- If app drawer swipe fails, look for a 'Settings' app icon directly"
                generator_message += f"\n- If quick settings swipe fails, try long-pressing status bar elements"
                generator_message += f"\n- Consider using voice commands or search functionality"
                generator_message += f"\n- Try opening apps directly with {{\"action_type\": \"open_app\", \"app_name\": \"Settings\"}}"
                generator_message += f"\n- Look for alternative UI elements with similar names"
                generator_message += f"\n- Check if the task is already completed (look for enabled/disabled states)"
                generator_message += f"\n- Try different navigation methods (back, home, recent apps)"
            else:
                # Add general guidance for better exploration
                generator_message += f"\nEXPLORATION TIPS:"
                generator_message += f"\n- Look carefully at all available UI elements before giving up"
                generator_message += f"\n- Check if the desired state is already achieved"
                generator_message += f"\n- Try different approaches if one doesn't work"
                generator_message += f"\n- Use {{\"action_type\": \"open_app\", \"app_name\": \"Settings\"}} to directly open apps"
                generator_message += f"\n- Look for UI elements with similar names (e.g., 'Internet' = 'WiFi')"
                generator_message += f"\n- Consider scrolling to see more options"
                generator_message += f"\n- Try long-pressing elements for additional options"
                generator_message += f"\n- **FIELD TYPES**: Look for `hint_text` to identify correct input fields"
                generator_message += f"\n- **DROPDOWNS**: If you see options like 'Home/Mobile/Work', click to select, don't type"
                generator_message += f"\n- **NAVIGATION**: If stuck, try {{\"action_type\": \"navigate_back\"}} to go back"
                generator_message += f"\n- **FIELD MATCHING**: Ensure data type matches field type (phone number → phone field)"

        # Only provide subinfo in the very first message
        if self.turn_count == 0:
            generator_message += f"Remember only complete the subtask: {subtask}\n"
            generator_message += f"You can use this extra information for completing the current subtask: {subtask_info}.\n"
            generator_message += device_context
        else:
            generator_message += device_context

        self.generator_agent.add_message(
            generator_message, image_content=obs.get("screenshot"), role="user"
        )

        plan = call_llm_safe(self.generator_agent)
        self.planner_history.append(plan)
        logger.info("PLAN: %s", plan)
        self.generator_agent.add_message(plan, role="assistant")

        # Show worker execution summary
        print(f"\n🔧 WORKER EXECUTION (Step {self.turn_count + 1}):")
        print(f"   Subtask: {subtask}")
        print(f"   Plan: {plan}")
        if self.turn_count > 0:
            print(f"   Previous Actions: {len(self.planner_history) - 1} actions taken")
        if screen_changed:
            print(f"   ✅ Screen changed detected")
        elif self.turn_count > 0:
            print(f"   ⚠️  No screen change detected - consider different approach")

        # Calculate input/output tokens and cost
        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)
        self.cost_this_turn += cost
        logger.info("WORKER COST: %s", self.cost_this_turn)

        # Parse and execute the action
        try:
            # Try to extract JSON action first (like T3A)
            import json
            import re
            import ast
            from typing import Any
            
            def extract_json(s: str) -> dict[str, Any] | None:
                """Extracts JSON from string like android_world agent_utils."""
                pattern = r'\{.*?\}'
                match = re.search(pattern, s, re.DOTALL)
                if match:
                    try:
                        return ast.literal_eval(match.group())
                    except (SyntaxError, ValueError) as error:
                        try:
                            # Try conversion with json module.
                            return json.loads(match.group())
                        except (SyntaxError, ValueError) as error2:
                            print(f'Cannot extract JSON, skipping due to errors {error} and {error2}')
                            return None
                else:
                    return None
            
            def parse_reason_action_output(raw_output: str) -> tuple[Optional[str], Optional[str]]:
                """Parses LLM action reason output like android_world m3a_utils."""
                reason_result = re.search(r'Reason:(.*)Action:', raw_output, flags=re.DOTALL)
                reason = reason_result.group(1).strip() if reason_result else None
                action_result = re.search(r'Action:(.*)', raw_output, flags=re.DOTALL)
                action = action_result.group(1).strip() if action_result else None
                if action:
                    extracted = extract_json(action)
                    if extracted is not None:
                        action = json.dumps(extracted)
                return reason, action
            
            # Parse the plan using T3A's exact approach
            reason, action = parse_reason_action_output(plan)
            
            if reason and action:
                try:
                    # Extract JSON using T3A's exact method
                    action_dict = extract_json(action)
                    if action_dict:
                        print(f"   Action: {action_dict}")
                        
                        # Create JSONAction like T3A does
                        from android_world.env.json_action import JSONAction
                        converted_action = JSONAction(**action_dict)
                        
                        # Execute directly like T3A does
                        exec_code = self.grounding_agent.env.execute_action(converted_action)
                        if exec_code is None:
                            exec_code = "SUCCESS: Action executed successfully"
                        
                        # Wait after navigation actions to allow UI to load
                        if action_dict.get('action_type') in ['click', 'open_app']:
                            import time
                            time.sleep(1.5)  # Wait for navigation to complete
                        
                        # For input_text actions, also press Enter like android_world does
                        if action_dict.get('action_type') == 'input_text':
                            try:
                                from android_world.env.json_action import KEYBOARD_ENTER
                                enter_action = JSONAction(action_type=KEYBOARD_ENTER)
                                enter_result = self.grounding_agent.env.execute_action(enter_action)
                            except Exception as e:
                                pass
                        
                    else:
                        logger.error("Failed to extract JSON from action")
                        exec_code = "FAIL: Could not parse JSON action"
                        
                except Exception as e:
                    logger.error(f"Error executing action: {e}")
                    exec_code = f"FAIL: Error executing action - {e}"
                    
            else:
                # No fallback needed - T3A approach should handle all cases
                logger.error("Could not parse reason and action from plan")
                exec_code = "FAIL: Could not parse plan output"
            
        except Exception as e:
            logger.error(f"Error in parsing/executing plan code: {e}")
            exec_code = self.grounding_agent.wait(1.0)
            # Always show execution error
            print(f"   ❌ Execution Error: {e}")
        
        executor_info = {
            "current_subtask": subtask,
            "current_subtask_info": subtask_info,
            "executor_plan": plan,
            "reflection": reflection,
            "num_input_tokens_executor": input_tokens,
            "num_output_tokens_executor": output_tokens,
            "screen_changed": screen_changed,
        }
        self.turn_count += 1

        if obs.get("screenshot"):
            self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()

        # Return the action dictionary for proper completion detection
        if reason and action:
            try:
                action_dict = extract_json(action)
                if action_dict:
                    return executor_info, [action_dict]
                else:
                    return executor_info, [exec_code]
            except:
                return executor_info, [exec_code]
        else:
            return executor_info, [exec_code]

    def clean_worker_generation_for_reflection(self, worker_generation: str) -> str:
        # Remove the previous action verification
        res = worker_generation[worker_generation.find("(UI Analysis)") :]
        # Extract the action from the JSON format
        action_match = re.search(r'Action:\s*(\{.*?\})', worker_generation, re.DOTALL)
        if action_match:
            action = action_match.group(1)
        else:
            action = "{}"
        # Cut off extra grounded actions
        res = res[: res.find("(Grounded Action)")]
        res += f"(Grounded Action)\n```json\n{action}\n```\n"
        return res
