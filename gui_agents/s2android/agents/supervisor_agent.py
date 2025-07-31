import logging
import textwrap
import time
from typing import Dict, Any, Optional, List, Tuple
from gui_agents.s2android.memory.procedural_memory_android import PROCEDURAL_MEMORY_ANDROID
from gui_agents.s2.utils.common_utils import call_llm_safe
from gui_agents.s2.core.mllm import LMMAgent

logger = logging.getLogger(__name__)


class AndroidSupervisorAgent:
    """Android-specific supervisor agent that reviews entire episodes and suggests improvements"""
    
    def __init__(self, engine_params: Dict):
        """Initialize the Android supervisor agent
        
        Args:
            engine_params: Configuration parameters for the LLM engine
        """
        self.engine_params = engine_params
        self.supervisor_engine = LMMAgent(engine_params)
        self.reset()
    
    def reset(self):
        """Reset the supervisor agent state"""
        self.review_history = []
        
    def review_episode(
        self,
        task_description: str,
        episode_frames: List[Dict],
        episode_actions: List[Dict],
        final_success: bool
    ) -> Tuple[str, Dict]:
        """
        Review an entire episode and provide improvement suggestions
        
        Args:
            task_description: The original task description
            episode_frames: List of frame data containing UI images and state
            episode_actions: List of actions taken during the episode
            final_success: Whether the task was ultimately successful
            
        Returns:
            Tuple of (review_summary, review_details)
        """
        
        # Save UI images for analysis
        saved_images = self._save_episode_images(episode_frames)
        
        # Construct review prompt with image references
        review_prompt = self._construct_review_prompt(
            task_description, episode_frames, episode_actions, final_success, saved_images
        )
        
        # Add to supervisor engine with images
        self.supervisor_engine.add_message(
            text_content=review_prompt,
            role="user"
        )
        
        # Get review response
        review_response = call_llm_safe(self.supervisor_engine)
        
        # Parse the response
        review_summary, review_details = self._parse_review_response(review_response)
        
        # Store in history
        review_info = {
            "task_description": task_description,
            "episode_frames": episode_frames,
            "episode_actions": episode_actions,
            "final_success": final_success,
            "saved_images": saved_images,
            "review_response": review_response,
            "review_summary": review_summary,
            "review_details": review_details
        }
        self.review_history.append(review_info)
        
        return review_summary, review_details
    
    def _save_episode_images(self, episode_frames: List[Dict]) -> List[str]:
        """Save UI images from episode frames for analysis"""
        import os
        import base64
        from PIL import Image
        import io
        
        saved_images = []
        timestamp = int(time.time())
        
        # Create images directory if it doesn't exist
        images_dir = f"episode_images_{timestamp}"
        os.makedirs(images_dir, exist_ok=True)
        
        for i, frame in enumerate(episode_frames):
            screenshot = frame.get('screenshot')
            if screenshot:
                try:
                    # Convert base64 to image and save
                    if isinstance(screenshot, str) and screenshot.startswith('data:image'):
                        # Handle base64 data URL
                        image_data = screenshot.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                    elif isinstance(screenshot, str):
                        # Handle base64 string
                        image_bytes = base64.b64decode(screenshot)
                    else:
                        # Handle bytes directly
                        image_bytes = screenshot
                    
                    # Save image
                    image_path = os.path.join(images_dir, f"frame_{i+1:03d}.png")
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    saved_images.append(image_path)
                    print(f"   📸 Saved frame {i+1} to {image_path}")
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to save frame {i+1}: {e}")
        
        return saved_images
    
    def _construct_review_prompt(
        self,
        task_description: str,
        episode_frames: List[Dict],
        episode_actions: List[Dict],
        final_success: bool,
        saved_images: List[str]
    ) -> str:
        """Construct the episode review prompt"""
        
        # Format episode summary
        num_frames = len(episode_frames)
        num_actions = len(episode_actions)
        
        # Extract key information from frames with detailed UI analysis
        frame_summaries = []
        for i, frame in enumerate(episode_frames):
            activity = frame.get('device_info', {}).get('current_activity', 'Unknown')
            ui_elements = frame.get('ui_elements', [])
            num_elements = len(ui_elements) if ui_elements else 0
            
            # Get detailed UI elements for analysis
            key_elements = []
            input_fields = []
            dropdowns = []
            buttons = []
            
            if ui_elements:
                for j, element in enumerate(ui_elements[:10]):  # Show first 10 elements
                    element_text = getattr(element, 'text', '')
                    element_hint = getattr(element, 'hint_text', '')
                    element_class = getattr(element, 'class_name', '')
                    element_editable = getattr(element, 'is_editable', False)
                    element_clickable = getattr(element, 'is_clickable', False)
                    
                    element_desc = f"Element {j}: '{element_text}'"
                    if element_hint:
                        element_desc += f" [hint: '{element_hint}']"
                    if element_editable:
                        element_desc += " [editable]"
                    element_desc += f" ({element_class})"
                    
                    key_elements.append(element_desc)
                    
                    # Categorize elements for analysis
                    if element_editable:
                        input_fields.append(element_desc)
                    elif element_class == 'Spinner':
                        dropdowns.append(element_desc)
                    elif element_clickable and element_text:
                        buttons.append(element_desc)
            
            frame_summary = f"Frame {i+1}: Activity={activity}, UI Elements={num_elements}"
            if key_elements:
                frame_summary += f"\n   Key Elements: {', '.join(key_elements[:5])}"
            if input_fields:
                frame_summary += f"\n   Input Fields: {', '.join(input_fields[:3])}"
            if dropdowns:
                frame_summary += f"\n   Dropdowns: {', '.join(dropdowns[:3])}"
            if buttons:
                frame_summary += f"\n   Buttons: {', '.join(buttons[:3])}"
            
            frame_summaries.append(frame_summary)
        
        # Format action summary
        action_summaries = []
        for i, action in enumerate(episode_actions):
            action_type = action.get('action_type', 'unknown')
            action_index = action.get('index', 'N/A')
            action_text = action.get('text', '')
            
            action_summary = f"Action {i+1}: {action_type}"
            if action_index != 'N/A':
                action_summary += f" (index: {action_index})"
            if action_text:
                action_summary += f" (text: '{action_text}')"
            
            action_summaries.append(action_summary)
        
        # Construct the review prompt with image analysis
        image_info = ""
        if saved_images:
            image_info = f"\nUI IMAGES SAVED: {len(saved_images)} images available for analysis"
            image_info += f"\nImage paths: {', '.join(saved_images)}"
        
        review_prompt = f"""
You are an expert Android UI automation supervisor. Review the following episode and provide detailed analysis and improvement suggestions based on the actual UI elements and actions observed.

TASK: {task_description}
FINAL SUCCESS: {'✅ SUCCESS' if final_success else '❌ FAILURE'}

EPISODE SUMMARY:
- Total Frames: {num_frames}
- Total Actions: {num_actions}
- Success Rate: {'100%' if final_success else '0%'}
{image_info}

DETAILED FRAME-BY-FRAME ANALYSIS:
{chr(10).join(frame_summaries)}

ACTION-BY-ACTION ANALYSIS:
{chr(10).join(action_summaries)}

SPECIFIC ANALYSIS REQUIREMENTS:
1. **Field Identification Analysis**: 
   - Did the agent correctly identify input fields vs dropdowns?
   - Did it use hint_text to identify field purposes?
   - Did it avoid typing in dropdown fields?
   - Did it enter data in the correct field types?

2. **UI Element Recognition**:
   - Did the agent find and interact with the correct UI elements?
   - Did it use the right element indices?
   - Did it handle editable vs non-editable elements correctly?

3. **Navigation Efficiency**:
   - Were there unnecessary navigation steps?
   - Did the agent get stuck or confused?
   - Could it have used more direct approaches?

4. **Action Sequence Analysis**:
   - Were actions taken in the optimal order?
   - Were there redundant or failed actions?
   - Did the agent handle errors gracefully?

5. **Specific UI Issues**:
   - Field confusion (phone number in company field, etc.)
   - Dropdown vs input field confusion
   - Wrong element selection
   - Missing alternative approaches

Provide your analysis in the following format:

**EPISODE REVIEW SUMMARY**
[Brief overview based on actual UI elements and actions observed]

**KEY ISSUES IDENTIFIED**
[List the main problems found]

**IMPROVEMENT SUGGESTIONS**
[Specific, actionable suggestions for improvement]

**BEST PRACTICES RECOMMENDATIONS**
[General guidelines for better performance]

**TECHNICAL RECOMMENDATIONS**
[Specific technical improvements for the agent system]
"""
        
        return review_prompt
    
    def _parse_review_response(self, response: str) -> Tuple[str, Dict]:
        """Parse the supervisor review response"""
        
        # Extract sections from the response
        sections = {
            "summary": "",
            "issues": "",
            "improvements": "",
            "best_practices": "",
            "technical_recommendations": ""
        }
        
        current_section = "summary"
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if "**EPISODE REVIEW SUMMARY**" in line:
                current_section = "summary"
                continue
            elif "**KEY ISSUES IDENTIFIED**" in line:
                current_section = "issues"
                continue
            elif "**IMPROVEMENT SUGGESTIONS**" in line:
                current_section = "improvements"
                continue
            elif "**BEST PRACTICES RECOMMENDATIONS**" in line:
                current_section = "best_practices"
                continue
            elif "**TECHNICAL RECOMMENDATIONS**" in line:
                current_section = "technical_recommendations"
                continue
            elif line.startswith("**") and line.endswith("**"):
                # Skip other bold headers
                continue
            else:
                # Add line to current section
                if sections[current_section]:
                    sections[current_section] += "\n" + line
                else:
                    sections[current_section] = line
        
        # Create review summary (first 200 characters of summary)
        review_summary = sections["summary"][:200] + "..." if len(sections["summary"]) > 200 else sections["summary"]
        
        # Create review details
        review_details = {
            "summary": sections["summary"],
            "issues": sections["issues"],
            "improvements": sections["improvements"],
            "best_practices": sections["best_practices"],
            "technical_recommendations": sections["technical_recommendations"]
        }
        
        return review_summary, review_details
    
    def get_review_summary(self) -> Dict:
        """Get a summary of all reviews performed"""
        if not self.review_history:
            return {"message": "No reviews performed yet"}
        
        total_reviews = len(self.review_history)
        successful_episodes = sum(1 for review in self.review_history if review["final_success"])
        success_rate = (successful_episodes / total_reviews) * 100 if total_reviews > 0 else 0
        
        return {
            "total_reviews": total_reviews,
            "successful_episodes": successful_episodes,
            "success_rate": f"{success_rate:.1f}%",
            "latest_review": self.review_history[-1] if self.review_history else None
        }
    
    def print_review(self, review_summary: str, review_details: Dict):
        """Print a formatted review"""
        print("\n" + "="*60)
        print("🔍 SUPERVISOR EPISODE REVIEW")
        print("="*60)
        print(f"📋 Summary: {review_summary}")
        print("\n" + "-"*40)
        
        if review_details.get("issues"):
            print("❌ KEY ISSUES IDENTIFIED:")
            print(review_details["issues"])
            print("\n" + "-"*40)
        
        if review_details.get("improvements"):
            print("💡 IMPROVEMENT SUGGESTIONS:")
            print(review_details["improvements"])
            print("\n" + "-"*40)
        
        if review_details.get("best_practices"):
            print("📚 BEST PRACTICES RECOMMENDATIONS:")
            print(review_details["best_practices"])
            print("\n" + "-"*40)
        
        if review_details.get("technical_recommendations"):
            print("⚙️  TECHNICAL RECOMMENDATIONS:")
            print(review_details["technical_recommendations"])
        
        print("="*60) 