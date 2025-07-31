import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from gui_agents.s2.utils.common_utils import (
    call_llm_safe,
    parse_single_code_from_string,
)

# Add android_world to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
android_world_path = os.path.join(project_root, 'android_world')
sys.path.insert(0, android_world_path)

from android_world.env.interface import AsyncAndroidEnv
from android_world.env.json_action import JSONAction, CLICK, INPUT_TEXT, SCROLL, LONG_PRESS, SWIPE, ANSWER, WAIT, OPEN_APP

logger = logging.getLogger("androidenv.grounding")


class ACI:
    def __init__(self):
        self.notes: List[str] = []


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


class AndroidACI(ACI):
    """Android-specific ACI that handles actual Android environment interactions"""
    
    def __init__(
        self,
        env: AsyncAndroidEnv,
    ):
        """Initialize Android ACI
        
        Args:
            env: The Android environment interface
        """
        super().__init__()
        
        self.env = env

    @agent_action
    def click(self, element_description: str):
        """Click on an Android UI element by description"""
        try:
            # Get current state to find element
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            # Find element by description
            element_index = self._find_element_by_description(element_description, ui_elements)
            if element_index is None:
                return f"FAIL: Could not find element '{element_description}'"
            
            action = JSONAction(action_type=CLICK, index=element_index)
            self.env.execute_action(action)
            return f"SUCCESS: Clicked element '{element_description}' (index {element_index})"
        except Exception as e:
            logger.error(f"Error executing click action: {e}")
            return f"FAIL: Error clicking element '{element_description}'"

    @agent_action
    def type(self, element_description: str, text: str):
        """Type text into an Android UI element by description"""
        try:
            # Get current state to find element
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            # Find element by description
            element_index = self._find_element_by_description(element_description, ui_elements)
            if element_index is None:
                return f"FAIL: Could not find element '{element_description}'"
            
            action = JSONAction(action_type=INPUT_TEXT, index=element_index, text=text)
            self.env.execute_action(action)
            return f"SUCCESS: Typed '{text}' into element '{element_description}' (index {element_index})"
        except Exception as e:
            logger.error(f"Error executing type action: {e}")
            return f"FAIL: Error typing into element '{element_description}'"

    @agent_action
    def scroll(self, direction: str, element_description: str = None):
        """Scroll in the specified direction (optionally on a specific element)"""
        try:
            action = JSONAction(action_type=SCROLL, direction=direction)
            
            # If element description is provided, find the element
            if element_description:
                state = self.env.get_state()
                ui_elements = state.ui_elements
                element_index = self._find_element_by_description(element_description, ui_elements)
                if element_index is None:
                    return f"FAIL: Could not find element '{element_description}'"
                action.index = element_index
                return f"SUCCESS: Scrolled {direction} on element '{element_description}' (index {element_index})"
            else:
                # Scroll the whole screen
                self.env.execute_action(action)
                return f"SUCCESS: Scrolled {direction} on screen"
        except Exception as e:
            logger.error(f"Error executing scroll action: {e}")
            return f"FAIL: Error scrolling {direction}"

    @agent_action
    def long_click(self, element_description: str, duration: float = 1.0):
        """Long click on an Android UI element"""
        try:
            # Get current state to find element
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            # Find element by description
            element_index = self._find_element_by_description(element_description, ui_elements)
            if element_index is None:
                return f"FAIL: Could not find element '{element_description}'"
            
            action = JSONAction(action_type=LONG_PRESS, index=element_index)
            self.env.execute_action(action)
            return f"SUCCESS: Long clicked element '{element_description}' (index {element_index}) for {duration}s"
        except Exception as e:
            logger.error(f"Error executing long click action: {e}")
            return f"FAIL: Error long clicking element '{element_description}'"

    @agent_action
    def swipe(self, direction: str, duration: float = 1.0):
        """Swipe in the specified direction with Android-specific handling"""
        try:
            # Normalize direction
            direction = direction.lower().strip()
            
            # Use JSONAction like other methods
            action = JSONAction(action_type=SWIPE, direction=direction)
            
            # Add duration if supported
            if hasattr(action, 'duration'):
                action.duration = duration
            
            self.env.execute_action(action)
            return f"SUCCESS: Swiped {direction}"
                
        except Exception as e:
            logger.error(f"Error executing swipe action: {e}")
            return f"FAIL: Error swiping {direction}"

    @agent_action
    def wait(self, time: float):
        """Wait for a specified amount of time"""
        import time as time_module
        time_module.sleep(time)
        return f"SUCCESS: Waited {time} seconds"

    @agent_action
    def done(self, return_value: Optional[Union[Dict, str, List, Tuple, int, float, bool]] = None):
        """End the current task with success"""
        self.returned_info = return_value
        return "DONE"

    @agent_action
    def fail(self):
        """End the current task with failure"""
        return "FAIL"

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """Save information to knowledge base"""
        self.notes.extend(text)
        return "SUCCESS: Information saved to knowledge base"

    @agent_action
    def is_app_drawer_open(self):
        """Check if the app drawer is currently open"""
        try:
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            # Look for app drawer indicators
            app_drawer_indicators = [
                "all apps", "app drawer", "apps", "applications",
                "search apps", "app library", "app grid"
            ]
            
            for element in ui_elements:
                if hasattr(element, 'text') and element.text:
                    text_lower = element.text.lower()
                    for indicator in app_drawer_indicators:
                        if indicator in text_lower:
                            return f"SUCCESS: App drawer is open (found '{element.text}')"
            
            return f"FAIL: App drawer is not open (no indicators found)"
        except Exception as e:
            logger.error(f"Error checking app drawer status: {e}")
            return f"FAIL: Error checking app drawer status"

    @agent_action
    def is_quick_settings_open(self):
        """Check if the quick settings panel is currently open"""
        try:
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            # Look for quick settings indicators
            quick_settings_indicators = [
                "wifi", "bluetooth", "airplane mode", "mobile data",
                "do not disturb", "auto rotate", "flashlight", "battery saver"
            ]
            
            for element in ui_elements:
                if hasattr(element, 'text') and element.text:
                    text_lower = element.text.lower()
                    for indicator in quick_settings_indicators:
                        if indicator in text_lower:
                            return f"SUCCESS: Quick settings is open (found '{element.text}')"
            
            return f"FAIL: Quick settings is not open (no indicators found)"
        except Exception as e:
            logger.error(f"Error checking quick settings status: {e}")
            return f"FAIL: Error checking quick settings status"

    def _find_element_by_description(self, description: str, ui_elements: List[Any]) -> Optional[int]:
        """Find UI element by description with synonym support"""
        if not description or not ui_elements:
            return None
            
        description_lower = description.lower().strip()
        
        # Define synonyms for common UI elements
        synonyms = {
            "wifi": ["internet", "network", "wireless", "wi-fi", "wifi settings"],
            "internet": ["wifi", "network", "wireless", "wi-fi", "wifi settings"],
            "settings": ["system preferences", "system settings", "preferences", "gear", "cog"],
            "home": ["home screen", "main screen", "desktop"],
            "back": ["back button", "back arrow", "return"],
            "close": ["x", "exit", "cancel", "dismiss"],
            "ok": ["okay", "confirm", "yes", "accept"],
            "cancel": ["no", "abort", "dismiss"],
            "done": ["finish", "complete", "save"],
            "save": ["done", "finish", "complete"],
            "search": ["find", "lookup", "magnifying glass"],
            "menu": ["hamburger", "three lines", "options"],
            "more": ["three dots", "ellipsis", "options"],
            "add": ["plus", "+", "new", "create"],
            "delete": ["trash", "remove", "minus", "-"],
            "edit": ["pencil", "modify", "change"],
            "share": ["send", "export", "forward"],
            "download": ["save", "get", "install"],
            "upload": ["send", "import", "attach"],
            "refresh": ["reload", "sync", "update"],
            "filter": ["sort", "organize", "arrange"],
            "sort": ["filter", "organize", "arrange"],
            "favorite": ["star", "like", "bookmark"],
            "bookmark": ["favorite", "star", "save"],
            "notifications": ["alerts", "messages", "bell"],
            "quick settings": ["quick panel", "control center", "settings panel"],
            "app drawer": ["all apps", "apps", "applications"],
            "recent apps": ["recent", "multitask", "switch apps"],
            "volume": ["sound", "audio", "speaker"],
            "brightness": ["light", "display", "screen"],
            "battery": ["power", "charge", "energy"],
            "data": ["mobile data", "cellular", "network"],
            "bluetooth": ["bt", "wireless", "connect"],
            "location": ["gps", "maps", "where"],
            "camera": ["photo", "picture", "video"],
            "gallery": ["photos", "images", "pictures"],
            "phone": ["call", "dialer", "contacts"],
            "messages": ["sms", "text", "chat"],
            "email": ["mail", "gmail", "outlook"],
            "calendar": ["schedule", "events", "dates"],
            "clock": ["time", "alarm", "timer"],
            "calculator": ["calc", "math", "compute"],
            "maps": ["navigation", "location", "directions"],
            "browser": ["chrome", "web", "internet"],
            "play store": ["google play", "apps", "install"],
            "files": ["documents", "storage", "my files"],
            "music": ["audio", "songs", "playlist"],
            "video": ["movies", "youtube", "media"],
            "games": ["play", "entertainment", "fun"],
            "health": ["fitness", "wellness", "activity"],
            "banking": ["finance", "money", "payments"],
            "shopping": ["store", "buy", "purchase"],
            "travel": ["trip", "booking", "vacation"],
            "food": ["restaurant", "delivery", "dining"],
            "transport": ["uber", "lyft", "taxi"],
            "social": ["facebook", "instagram", "twitter"],
            "work": ["office", "productivity", "business"],
            "education": ["learning", "study", "school"],
            "news": ["information", "articles", "updates"],
            "weather": ["forecast", "temperature", "climate"],
            "notes": ["memo", "text", "writing"],
            "voice": ["recorder", "audio", "speech"],
            "translate": ["language", "dictionary", "words"],
            "scan": ["qr", "barcode", "document"],
            "password": ["security", "lock", "key"],
            "backup": ["cloud", "sync", "restore"],
            "update": ["upgrade", "install", "download"],
            "reset": ["factory", "clear", "wipe"],
            "developer": ["debug", "advanced", "technical"],
            "accessibility": ["assistive", "help", "support"],
            "privacy": ["security", "permissions", "data"],
            "storage": ["memory", "space", "files"],
            "performance": ["speed", "optimization", "battery"],
            "display": ["screen", "resolution", "brightness"],
            "sound": ["audio", "volume", "vibration"],
            "language": ["locale", "region", "translation"],
            "date": ["time", "calendar", "schedule"],
            "about": ["info", "version", "device"],
            "help": ["support", "assistance", "guide"],
            "feedback": ["report", "suggest", "comment"],
            "terms": ["legal", "agreement", "policy"],
            "privacy policy": ["data", "information", "rights"],
            "licenses": ["legal", "copyright", "attribution"],
            "version": ["build", "release", "update"],
            "build": ["version", "number", "release"],
            "model": ["device", "phone", "hardware"],
            "serial": ["imei", "number", "identifier"],
            "android": ["version", "api", "level"],
            "kernel": ["linux", "system", "core"],
            "baseband": ["radio", "modem", "cellular"],
            "build number": ["version", "release", "number"],
            "security patch": ["update", "fix", "vulnerability"],
            "google play": ["play store", "apps", "install"],
            "system webview": ["browser", "web", "internet"],
            "android system": ["core", "system", "framework"],
            "google services": ["play", "framework", "services"],
            "carrier services": ["mobile", "network", "cellular"],
            "digital wellbeing": ["screen time", "usage", "balance"],
            "device health": ["battery", "storage", "performance"],
            "find my device": ["location", "tracking", "security"],
            "google": ["search", "assistant", "services"],
            "chrome": ["browser", "web", "internet"],
            "gmail": ["email", "mail", "messages"],
            "youtube": ["video", "media", "entertainment"],
            "maps": ["navigation", "location", "directions"],
            "drive": ["storage", "cloud", "files"],
            "photos": ["gallery", "images", "pictures"],
            "calendar": ["schedule", "events", "dates"],
            "contacts": ["people", "phone", "addresses"],
            "phone": ["dialer", "call", "contacts"],
            "messages": ["sms", "text", "chat"],
            "clock": ["time", "alarm", "timer"],
            "calculator": ["calc", "math", "compute"],
            "camera": ["photo", "picture", "video"],
            "gallery": ["photos", "images", "pictures"],
            "music": ["audio", "songs", "playlist"],
            "video": ["movies", "youtube", "media"],
            "games": ["play", "entertainment", "fun"],
            "files": ["documents", "storage", "my files"],
            "notes": ["memo", "text", "writing"],
            "voice recorder": ["audio", "speech", "recording"],
            "translate": ["language", "dictionary", "words"],
            "scan": ["qr", "barcode", "document"],
            "weather": ["forecast", "temperature", "climate"],
            "news": ["information", "articles", "updates"],
            "health": ["fitness", "wellness", "activity"],
            "banking": ["finance", "money", "payments"],
            "shopping": ["store", "buy", "purchase"],
            "travel": ["trip", "booking", "vacation"],
            "food": ["restaurant", "delivery", "dining"],
            "transport": ["uber", "lyft", "taxi"],
            "social": ["facebook", "instagram", "twitter"],
            "work": ["office", "productivity", "business"],
            "education": ["learning", "study", "school"],
            "entertainment": ["media", "fun", "leisure"],
            "utilities": ["tools", "system", "maintenance"],
            "productivity": ["work", "office", "business"],
            "communication": ["social", "messaging", "email"],
            "lifestyle": ["health", "fitness", "wellness"],
            "reference": ["dictionary", "encyclopedia", "learning"],
            "sports": ["fitness", "exercise", "activity"],
            "medical": ["health", "wellness", "fitness"],
            "finance": ["banking", "money", "payments"],
            "navigation": ["maps", "location", "directions"],
            "photography": ["camera", "photos", "images"],
            "video & audio": ["media", "entertainment", "streaming"],
            "books & reference": ["reading", "learning", "education"],
            "business": ["work", "office", "productivity"],
            "comics": ["reading", "entertainment", "media"],
            "communication": ["social", "messaging", "email"],
            "education": ["learning", "study", "school"],
            "entertainment": ["media", "fun", "leisure"],
            "family": ["kids", "parental", "children"],
            "game": ["play", "entertainment", "fun"],
            "health & fitness": ["wellness", "activity", "medical"],
            "libraries & demo": ["development", "testing", "demo"],
            "lifestyle": ["health", "fitness", "wellness"],
            "live wallpaper": ["customization", "personalization", "theme"],
            "media & video": ["entertainment", "streaming", "media"],
            "medical": ["health", "wellness", "fitness"],
            "music & audio": ["entertainment", "streaming", "media"],
            "news & magazines": ["information", "articles", "updates"],
            "personalization": ["customization", "theme", "appearance"],
            "photography": ["camera", "photos", "images"],
            "productivity": ["work", "office", "business"],
            "shopping": ["store", "buy", "purchase"],
            "social": ["communication", "messaging", "networking"],
            "sports": ["fitness", "exercise", "activity"],
            "tools": ["utilities", "system", "maintenance"],
            "transportation": ["travel", "navigation", "mobility"],
            "travel & local": ["navigation", "location", "directions"],
            "weather": ["forecast", "temperature", "climate"],
            "writing": ["notes", "text", "documentation"]
        }
        
        # Check exact match first
        for i, element in enumerate(ui_elements):
            # Safely get element attributes with null checks
            element_text = getattr(element, 'text', None)
            element_desc = getattr(element, 'content_description', None)
            
            # Convert to strings and handle None values
            element_text_str = str(element_text).lower() if element_text is not None else ""
            element_desc_str = str(element_desc).lower() if element_desc is not None else ""
            
            if (description_lower in element_text_str or 
                description_lower in element_desc_str or
                element_text_str in description_lower or
                element_desc_str in description_lower):
                    return i
        
        # Check synonyms if no exact match
        for synonym_group, synonym_list in synonyms.items():
            if description_lower in synonym_group or any(syn in description_lower for syn in synonym_list):
                for i, element in enumerate(ui_elements):
                    # Safely get element attributes with null checks
                    element_text = getattr(element, 'text', None)
                    element_desc = getattr(element, 'content_description', None)
                    
                    # Convert to strings and handle None values
                    element_text_str = str(element_text).lower() if element_text is not None else ""
                    element_desc_str = str(element_desc).lower() if element_desc is not None else ""
                    
                    # Check if element matches any synonym
                    for synonym in synonym_list:
                        if (synonym in element_text_str or 
                            synonym in element_desc_str or
                            element_text_str in synonym or
                            element_desc_str in synonym):
                                return i
            
                    # Also check the main synonym group
                    if (synonym_group in element_text_str or 
                        synonym_group in element_desc_str or
                        element_text_str in synonym_group or
                        element_desc_str in synonym_group):
                                return i
        
        return None

    @agent_action
    def click_by_index(self, index: int):
        """Click on a UI element by index"""
        try:
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            if index < 0 or index >= len(ui_elements):
                return f"FAIL: Index {index} out of range (0-{len(ui_elements)-1})"
            
            action = JSONAction(action_type=CLICK, index=index)
            self.env.execute_action(action)
            return f"SUCCESS: Clicked element at index {index}"
        except Exception as e:
            logger.error(f"Error executing click by index: {e}")
            return f"FAIL: Error clicking element at index {index}"

    @agent_action
    def long_click_by_index(self, index: int):
        """Long click on a UI element by index"""
        try:
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            if index < 0 or index >= len(ui_elements):
                return f"FAIL: Index {index} out of range (0-{len(ui_elements)-1})"
            
            action = JSONAction(action_type=LONG_PRESS, index=index)
            self.env.execute_action(action)
            return f"SUCCESS: Long clicked element at index {index}"
        except Exception as e:
            logger.error(f"Error executing long click by index: {e}")
            return f"FAIL: Error long clicking element at index {index}"

    @agent_action
    def type_by_index(self, index: int, text: str):
        """Type text into a UI element by index"""
        try:
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            if index < 0 or index >= len(ui_elements):
                return f"FAIL: Index {index} out of range (0-{len(ui_elements)-1})"
            
            action = JSONAction(action_type=INPUT_TEXT, index=index, text=text)
            print(text)
            self.env.execute_action(action)
            return f"SUCCESS: Typed '{text}' into element at index {index}"
        except Exception as e:
            logger.error(f"Error executing type by index: {e}")
            return f"FAIL: Error typing into element at index {index}"

    @agent_action
    def navigate_home(self):
        """Navigate to home screen"""
        try:
            # Use JSONAction for navigation
            action = JSONAction(action_type="navigate_home")
            self.env.execute_action(action)
            return "SUCCESS: Navigated to home"
        except Exception as e:
            logger.error(f"Error navigating home: {e}")
            return f"FAIL: Error navigating home"

    @agent_action
    def navigate_back(self):
        """Navigate back"""
        try:
            # Use JSONAction for navigation
            action = JSONAction(action_type="navigate_back")
            self.env.execute_action(action)
            return "SUCCESS: Navigated back"
        except Exception as e:
            logger.error(f"Error navigating back: {e}")
            return f"FAIL: Error navigating back"

    @agent_action
    def swipe_partial(self, direction: str):
        """Swipe in a direction using partial gesture"""
        try:
            direction = direction.lower().strip()
            
            # Get screen dimensions for partial swipes
            state = self.env.get_state()
            screen_width = state.screen_width
            screen_height = state.screen_height
            
            # Calculate partial swipe coordinates (like T3A)
            if direction == "up":
                # Partial swipe up from middle to top (for app drawer)
                start_x = screen_width // 2
                start_y = int(screen_height * 0.7)  # 70% down
                end_x = screen_width // 2
                end_y = int(screen_height * 0.3)    # 30% down
            elif direction == "down":
                # Partial swipe down from top to middle (for notifications)
                start_x = screen_width // 2
                start_y = int(screen_height * 0.1)  # 10% down
                end_x = screen_width // 2
                end_y = int(screen_height * 0.6)    # 60% down
            elif direction == "left":
                # Partial swipe left
                start_x = int(screen_width * 0.8)
                start_y = screen_height // 2
                end_x = int(screen_width * 0.2)
                end_y = screen_height // 2
            elif direction == "right":
                # Partial swipe right
                start_x = int(screen_width * 0.2)
                start_y = screen_height // 2
                end_x = int(screen_width * 0.8)
                end_y = screen_height // 2
            else:
                return f"FAIL: Invalid direction '{direction}'. Use up/down/left/right"
            
            # Use JSONAction with coordinates for precise control
            action = JSONAction(
                action_type=SWIPE,
                direction=direction,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                duration=500  # 500ms duration for smooth gesture
            )
            
            self.env.execute_action(action)
            return f"SUCCESS: Swiped {direction} (partial gesture)"
            
        except Exception as e:
            logger.error(f"Error executing partial swipe: {e}")
            return f"FAIL: Error swiping {direction}"

    @agent_action
    def open_app(self, app_name: str):
        """Open an app by name (like T3A)"""
        try:
            # Common app name mappings (like T3A's _PATTERN_TO_ACTIVITY)
            app_mappings = {
                "settings": "com.android.settings",
                "wifi": "com.android.settings",
                "bluetooth": "com.android.settings", 
                "camera": "com.android.camera",
                "gallery": "com.android.gallery3d",
                "phone": "com.android.dialer",
                "contacts": "com.android.contacts",
                "messages": "com.android.mms",
                "email": "com.android.email",
                "gmail": "com.google.android.gm",
                "chrome": "com.android.chrome",
                "browser": "com.android.browser",
                "calendar": "com.android.calendar",
                "clock": "com.android.deskclock",
                "calculator": "com.android.calculator2",
                "maps": "com.google.android.apps.maps",
                "youtube": "com.google.android.youtube",
                "play store": "com.android.vending",
                "files": "com.android.documentsui",
                "music": "com.android.music",
                "video": "com.android.gallery3d",
                "games": "com.android.games",
                "health": "com.google.android.apps.fitness",
                "banking": "com.android.banking",
                "shopping": "com.android.shopping",
                "travel": "com.android.travel",
                "food": "com.android.food",
                "transport": "com.android.transport",
                "social": "com.android.social",
                "work": "com.android.work",
                "education": "com.android.education",
                "news": "com.android.news",
                "weather": "com.android.weather",
                "notes": "com.android.notes",
                "voice": "com.android.voicerecorder",
                "translate": "com.google.android.apps.translate",
                "scan": "com.android.scanner",
                "password": "com.android.password",
                "backup": "com.android.backup",
                "update": "com.android.update",
                "reset": "com.android.reset",
                "developer": "com.android.developer",
                "accessibility": "com.android.accessibility",
                "privacy": "com.android.privacy",
                "storage": "com.android.storage",
                "performance": "com.android.performance",
                "display": "com.android.display",
                "sound": "com.android.sound",
                "language": "com.android.language",
                "date": "com.android.date",
                "about": "com.android.about",
                "help": "com.android.help",
                "feedback": "com.android.feedback",
                "terms": "com.android.terms",
                "privacy policy": "com.android.privacy",
                "licenses": "com.android.licenses",
                "version": "com.android.version",
                "build": "com.android.build",
                "model": "com.android.model",
                "serial": "com.android.serial",
                "android": "com.android.android",
                "kernel": "com.android.kernel",
                "baseband": "com.android.baseband",
                "build number": "com.android.build",
                "security patch": "com.android.security",
                "google play": "com.android.vending",
                "system webview": "com.android.webview",
                "android system": "com.android.system",
                "google services": "com.google.android.gms",
                "carrier services": "com.android.carrier",
                "digital wellbeing": "com.google.android.apps.wellbeing",
                "device health": "com.android.health",
                "find my device": "com.google.android.apps.findmydevice",
                "google": "com.google.android.google",
                "drive": "com.google.android.apps.docs",
                "photos": "com.google.android.apps.photos",
                "assistant": "com.google.android.apps.assistant",
                "home": "com.google.android.apps.nexuslauncher",
                "launcher": "com.google.android.apps.nexuslauncher"
            }
            
            # Try to find the app in the current UI first
            state = self.env.get_state()
            ui_elements = state.ui_elements
            
            # Look for app icon by name
            app_index = None
            for i, element in enumerate(ui_elements):
                # Safely get element attributes with null checks
                element_text = getattr(element, 'text', None)
                element_desc = getattr(element, 'content_description', None)
                
                # Convert to strings and handle None values
                element_text_str = str(element_text).lower() if element_text is not None else ""
                element_desc_str = str(element_desc).lower() if element_desc is not None else ""
                app_name_lower = app_name.lower()
                
                # Check if this element matches the app name
                if (app_name_lower in element_text_str or 
                    app_name_lower in element_desc_str or
                    element_text_str in app_name_lower or
                    element_desc_str in app_name_lower):
                    app_index = i
                    break
            
            if app_index is not None:
                # Click on the app icon
                action = JSONAction(action_type=CLICK, index=app_index)
                self.env.execute_action(action)
                return f"SUCCESS: Opened {app_name} by clicking icon"
            else:
                # Try to open app using system intent with mapped package name
                mapped_package = app_mappings.get(app_name.lower())
                if mapped_package:
                    # Try with mapped package name
                    action = JSONAction(action_type=OPEN_APP, app_name=mapped_package)
                    self.env.execute_action(action)
                    return f"SUCCESS: Opened {app_name} via system intent (mapped package)"
                else:
                    # Try with original app name
                    action = JSONAction(action_type=OPEN_APP, app_name=app_name)
                    self.env.execute_action(action)
                    return f"SUCCESS: Opened {app_name} via system intent"
                
        except Exception as e:
            logger.error(f"Error opening app {app_name}: {e}")
            return f"FAIL: Error opening app {app_name}"
