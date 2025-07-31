import logging
import textwrap
from typing import Dict, Any, Optional, Tuple
from gui_agents.s2android.memory.procedural_memory_android import PROCEDURAL_MEMORY_ANDROID
from gui_agents.s2.utils.common_utils import call_llm_safe
from gui_agents.s2.core.mllm import LMMAgent

logger = logging.getLogger(__name__)


class AndroidVerifierAgent:
    """Android-specific verifier agent that assesses execution progress and determines pass/fail status"""
    
    def __init__(self, engine_params: Dict):
        """Initialize the Android verifier agent
        
        Args:
            engine_params: Configuration parameters for the LLM engine
        """
        self.engine_params = engine_params
        self.verifier_engine = LMMAgent(engine_params)
        self.reset()
    
    def reset(self):
        """Reset the verifier agent state"""
        self.verification_history = []
        
    def verify_execution(
        self, 
        manager_goal: str,
        worker_execution_result: Dict,
        current_ui_state: Dict
    ) -> Tuple[bool, str, Dict]:
        """
        Verify if the worker execution was successful based on manager goal and current UI state
        
        Args:
            manager_goal: The goal/objective from the manager
            worker_execution_result: The result from worker execution
            current_ui_state: Current UI state observation
            
        Returns:
            Tuple of (is_success, reasoning, verification_info)
        """
        
        # Construct verification prompt
        verification_prompt = self._construct_verification_prompt(
            manager_goal, worker_execution_result, current_ui_state
        )
        
        # Add to verifier engine
        self.verifier_engine.add_message(
            text_content=verification_prompt,
            image_content=current_ui_state.get("screenshot"),
            role="user"
        )
        
        # Get verification response
        verification_response = call_llm_safe(self.verifier_engine)
        
        # Parse the response
        is_success, reasoning = self._parse_verification_response(verification_response)
        
        # Store in history
        verification_info = {
            "manager_goal": manager_goal,
            "worker_result": worker_execution_result,
            "verification_response": verification_response,
            "is_success": is_success,
            "reasoning": reasoning
        }
        self.verification_history.append(verification_info)
        
        return is_success, reasoning, verification_info
    
    def _construct_verification_prompt(
        self, 
        manager_goal: str, 
        worker_execution_result: Dict, 
        current_ui_state: Dict
    ) -> str:
        """Construct the verification prompt"""
        
        # Extract relevant information
        current_activity = current_ui_state.get('device_info', {}).get('current_activity', 'Unknown')
        ui_elements = current_ui_state.get('ui_elements', [])
        num_ui_elements = len(ui_elements) if ui_elements else 0
        
        # Get worker execution details
        worker_plan = worker_execution_result.get('executor_plan', '')
        worker_reflection = worker_execution_result.get('reflection', '')
        worker_actions = worker_execution_result.get('actions', [])
        screen_changed = worker_execution_result.get('screen_changed', False)
        
        # Format UI elements for analysis
        ui_summary = ""
        if ui_elements:
            element_descriptions = []
            for i, element in enumerate(ui_elements[:10]):  # Show first 10 elements
                element_text = getattr(element, 'text', '')
                element_desc = getattr(element, 'content_description', '')
                element_class = getattr(element, 'class_name', '')
                
                if element_text and element_desc:
                    element_descriptions.append(f"Element {i}: '{element_text}' (desc: '{element_desc}')")
                elif element_text:
                    element_descriptions.append(f"Element {i}: '{element_text}'")
                elif element_desc:
                    element_descriptions.append(f"Element {i}: (desc: '{element_desc}')")
                else:
                    element_descriptions.append(f"Element {i}: {element_class}")
            
            ui_summary = "\n".join(element_descriptions)
            if len(ui_elements) > 10:
                ui_summary += f"\n... and {len(ui_elements) - 10} more elements"
        
        prompt = textwrap.dedent(f"""
        You are an Android execution verifier. Your task is to assess whether the worker's execution was successful in achieving the manager's goal.
        
        **MANAGER GOAL:**
        {manager_goal}
        
        **WORKER EXECUTION RESULT:**
        - Plan: {worker_plan}
        - Reflection: {worker_reflection}
        - Actions Taken: {worker_actions}
        - Screen Changed: {screen_changed}
        
        **CURRENT UI STATE:**
        - Current Activity: {current_activity}
        - Number of UI Elements: {num_ui_elements}
        - UI Elements: {ui_summary}
        
        **VERIFICATION TASK:**
        Analyze whether the worker's execution successfully progressed toward the manager's goal.
        
        **EVALUATION CRITERIA:**
        1. **Progress Assessment**: Did the worker make meaningful progress toward the goal?
           - Successfully opening an app (activity changes)
           - Navigating through menus and sub-menus
           - Finding and clicking relevant UI elements
           - Successfully executing actions that change screen state
           - Making progress toward the target functionality (e.g., WiFi settings)
        
        2. **Action Effectiveness**: Were the actions appropriate and successful?
           - Actions were relevant to the goal
           - Actions were executed successfully
           - Actions led to expected UI changes
        
        3. **Goal Alignment**: Is the current state closer to achieving the manager's goal?
           - Current activity is relevant to the goal
           - Available UI elements support goal achievement
           - No major obstacles preventing goal completion
        
        **VERIFICATION OUTPUT:**
        Provide your assessment in the following format:
        
        VERIFICATION_RESULT: [PASS/FAIL]
        REASONING: [Detailed explanation of why the execution passed or failed]
        
        **IMPORTANT GUIDELINES:**
        - PASS if the worker made meaningful progress toward the goal
        - PASS if the worker is in the correct app/section for the goal
        - PASS if the worker successfully executed relevant actions
        - FAIL if the worker is stuck or making no progress
        - FAIL if the worker is in the wrong app/section
        - FAIL if the worker executed irrelevant or failed actions
        """)
        
        return prompt
    
    def _parse_verification_response(self, response: str) -> Tuple[bool, str]:
        """Parse the verification response to extract pass/fail status and reasoning"""
        
        # Extract verification result
        if "VERIFICATION_RESULT: PASS" in response.upper():
            is_success = True
        elif "VERIFICATION_RESULT: FAIL" in response.upper():
            is_success = False
        else:
            # Default to pass if unclear
            is_success = True
            logger.warning("Could not parse verification result, defaulting to PASS")
        
        # Extract reasoning
        reasoning_start = response.find("REASONING:")
        if reasoning_start != -1:
            reasoning = response[reasoning_start:].strip()
        else:
            reasoning = "No explicit reasoning provided"
        
        return is_success, reasoning
    
    def get_verification_summary(self) -> Dict:
        """Get a summary of verification history"""
        if not self.verification_history:
            return {"total_verifications": 0, "success_rate": 0.0}
        
        total = len(self.verification_history)
        successful = sum(1 for v in self.verification_history if v["is_success"])
        success_rate = successful / total if total > 0 else 0.0
        
        return {
            "total_verifications": total,
            "successful_verifications": successful,
            "failed_verifications": total - successful,
            "success_rate": success_rate,
            "recent_verifications": self.verification_history[-5:]  # Last 5 verifications
        } 