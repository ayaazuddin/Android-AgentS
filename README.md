# Android Agent-S2 Setup and Running Guide

This guide will help you set up and run the Android Agent-S2 system for automated Android UI interaction tasks.

## 📋 Prerequisites

### 1. System Requirements
- **Operating System**: Windows, macOS, or Linux
- **RAM**: At least 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **Python**: 3.11+ (required for android_world compatibility)

### 2. Required Software
- **Android Studio**: Latest version
- **Conda**: For Python environment management
- **Git**: For cloning repositories

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
# Clone the Agent-S repository
git clone 
cd Agent-S

# Initialize and update submodules
git submodule update --init --recursive
```

### Step 2: Set Up Android Emulator

#### 2.1 Install Android Studio
1. Download Android Studio from [https://developer.android.com/studio](https://developer.android.com/studio)
2. Install Android Studio following the installation wizard
3. Open Android Studio and complete the initial setup

#### 2.2 Create Android Virtual Device (AVD)
1. Open Android Studio
2. Go to **Tools** → **AVD Manager**
3. Click **Create Virtual Device**
4. Select **Pixel 6** for hardware
5. Choose **Tiramisu, API Level 33** for System Image
6. Name the AVD: **AndroidWorldAvd**
7. Click **Finish**

> **Note**: Watch the [setup video](https://example.com/setup-video) for detailed visual instructions.

#### 2.3 Launch Emulator from Command Line

**Important**: Launch the emulator from command line, NOT from Android Studio UI.

```bash
# Find your Android SDK path (usually in Android Studio settings)
# Common paths:
# Windows: C:\Users\username\AppData\Local\Android\Sdk
# macOS: ~/Library/Android/sdk
# Linux: ~/Android/Sdk

# Set ANDROID_HOME environment variable
export ANDROID_HOME=/path/to/your/Android/Sdk

# Launch the emulator with required flags
$ANDROID_HOME/emulator/emulator -avd AndroidWorldAvd -grpc 8554
```

> **Critical**: The `-grpc 8554` flag is required for communication with the accessibility forwarding app.

### Step 3: Set Up Python Environment

```bash
# Create conda environment with current Python version
conda create -n android_world python=3.11

# Activate the environment
conda activate android_world

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Step 4: Install Dependencies

#### 4.1 Install Base Requirements
```bash
# Navigate to the Agent-S directory
cd Agent-S

# Install base requirements
pip install -r requirements.txt
```

#### 4.2 Install Android World Dependencies
```bash
# Navigate to android_world directory
cd android_world

# Install android_world dependencies
pip install -e .
```

#### 4.3 Install Additional Dependencies
```bash
# Install additional packages needed for Android Agent-S2
pip install absl-py
pip install pillow
pip install opencv-python
pip install requests
pip install tiktoken
```

### Step 5: Install Android Agent-S2 Requirements

```bash
# Make sure you're in the android_world conda environment
conda activate android_world

# Install Android Agent-S2 specific requirements
pip install -r requirements_android_world.txt
```

### Step 6: Set Up Environment Variables

```bash
# Set OpenAI API key (required for LLM operations)
export OPENAI_API_KEY="your-openai-api-key-here"

# Set Android SDK path
export ANDROID_HOME="/path/to/your/Android/Sdk"

# Add platform-tools to PATH
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

## 🎯 Running the Android Agent-S2

### Prerequisites Check

Before running, ensure:
1. ✅ Android emulator is running with `-grpc 8554` flag
2. ✅ Conda environment `android_world` is activated
3. ✅ OpenAI API key is set
4. ✅ All dependencies are installed

### Basic Usage

```bash
# Navigate to the s2android directory
cd Agent-S/gui_agents/s2android

# Run with default settings (ContactsAddContact task)
python run_task.py

# Run with specific model
python run_task.py --model=gpt-4o-mini

# Run with specific task
python run_task.py --task=ContactsAddContact

# Run with custom API key
python run_task.py --api_key=your-api-key
```

### Advanced Usage

```bash
# Run with custom ADB path
python run_task.py --adb_path=/path/to/adb

# Run with different console port
python run_task.py --console_port=5556

# Run with emulator setup (only needed once)
python run_task.py --perform_emulator_setup=True

# Run with specific task and model
python run_task.py --task=ContactsAddContact --model=gpt-4o-mini
```

## 🔧 Troubleshooting

### Common Issues

#### 1. ADB Not Found
```bash
# Error: adb not found in common Android SDK paths
# Solution: Set correct ANDROID_HOME and PATH
export ANDROID_HOME="/path/to/your/Android/Sdk"
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

#### 2. Emulator Not Running
```bash
# Check if emulator is running
adb devices

# If no devices listed, restart emulator with correct flags
$ANDROID_HOME/emulator/emulator -avd AndroidWorldAvd -grpc 8554
```

#### 3. Import Errors
```bash
# If you get android_world import errors
cd Agent-S/android_world
pip install -e .
```

#### 4. Python Version Issues
```bash
# Ensure you're using Python 3.11+
python --version

# If not, recreate conda environment
conda deactivate
conda remove -n android_world --all
conda create -n android_world python=3.11
conda activate android_world
```

### Debug Mode

```bash
# Run with verbose logging
python run_task.py --verbosity=1

# Check emulator status
adb devices
adb shell getprop ro.build.version.release
```

## 📊 Expected Output

When running successfully, you should see:

```
🚀 Starting Android Agent-S2 Contact Task
📱 Using device on port: 5554
🤖 Using model: gpt-4o-mini
🔧 Setting up Android environment...
✅ Successfully imported android_world components
📋 Loading task registry...
🎯 Using default task: ContactsAddContact
🎯 Task Goal: Add a new contact with name 'John Doe' and phone number '555-1234'
🤖 Creating Android Agent-S2...
🔄 Starting task execution...
==================================================
🎯 Goal: Add a new contact with name 'John Doe' and phone number '555-1234'
==================================================

📝 Step 1/20
------------------------------
Worker → Add contact name: Enter the contact's first and last name
🔧 WORKER EXECUTION (Step 1):
   Subtask: Add contact name: Enter the contact's first and last name
   Plan: I need to add a contact name. Let me first open the contacts app...
   ✅ Screen changed detected
🔄 Action: {"action_type": "open_app", "app_name": "Contacts"}
```

## 🎯 Available Tasks

The system supports various Android tasks:

- **ContactsAddContact**: Add a new contact with name and phone number
- **ContactsEditContact**: Edit an existing contact
- **ContactsDeleteContact**: Delete a contact
- **SettingsWiFi**: Configure WiFi settings
- **SettingsBluetooth**: Configure Bluetooth settings
- **SettingsDisplay**: Configure display settings

## 🔍 Monitoring and Debugging

### Check Emulator Status
```bash
# List connected devices
adb devices

# Check emulator properties
adb shell getprop

# Check running processes
adb shell ps
```

### View Logs
```bash
# View system logs
adb logcat

# View specific app logs
adb logcat | grep "com.android.contacts"
```

### Debug Agent Behavior
```bash
# Run with detailed logging
python run_task.py --verbosity=2

# Check episode data collection
# Look for episode_images_* directories after running
```

## 📁 Project Structure

```
Agent-S/
├── gui_agents/
│   └── s2android/
│       ├── agents/           # Agent implementations
│       ├── memory/           # Procedural memory
│       ├── run_task.py       # Main execution script
│       └── README.md         # This file
├── android_world/            # Android World framework
├── requirements.txt          # Base requirements
└── README.md               # Main project README
```

## 🤝 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Verify all prerequisites** are met
3. **Check emulator is running** with correct flags
4. **Ensure conda environment** is activated
5. **Verify API key** is set correctly

For additional support, please check the main project README or create an issue in the repository.

## 📝 Notes

- The emulator must be launched with `-grpc 8554` flag for proper communication
- Python 3.11+ is required for android_world compatibility
- The system saves episode images for analysis in `episode_images_*` directories
- Supervisor agent provides detailed analysis of agent performance
- Field identification has been enhanced with hint_text analysis 