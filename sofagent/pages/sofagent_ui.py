import streamlit as st
import os
import sys
import base64 # For embedding images in HTML
from loguru import logger
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# --- PyTorch Workaround for Streamlit ---
# Add this BEFORE other significant imports that might use PyTorch
try:
    import torch
    if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = []
    print("Applied PyTorch __path__ workaround for Streamlit.") # Optional: for confirmation
except ImportError:
    print("PyTorch not found, skipping __path__ workaround.") # Optional
# --- End of Workaround ---

# Add project root to sys.path to allow imports from sofagent
# This is often needed when running streamlit apps from a subdirectory
# Adjust the number of ".." if your file structure is different or if you run it from project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sofagent.systems import CollaborationSystem
from sofagent.utils import init_api, read_json, add_chat_message as util_add_chat_message, get_avatar

# --- Configuration ---
API_CONFIG_PATH = "config/api-config.json"
# Use the main collaboration chat config
SYSTEM_CONFIG_PATH = "config/systems/collaboration/chat.json"
PAGE_TITLE = "SofAgent - Your Furniture Assistant"
# --- Define paths to your logos in the "images" directory (assumed to be in project root) ---
# Ensure you have an "images" folder in your project's root directory.
IMAGE_DIR = os.path.join(project_root, "images")
LOGO_PATH_1 = os.path.join(IMAGE_DIR, "Poliba logo.png") # Replace with your actual filename
LOGO_PATH_2 = os.path.join(IMAGE_DIR, "sisinflab_logo.png") # Replace with your actual filename
LOGO_PATH_3 = os.path.join(IMAGE_DIR, "Natuzzi-logo.png") # Replace with your actual filename
INITIAL_GREETING = "Ciao! Sono SofAgent, il tuo assistente per l'arredamento. Come posso aiutarti oggi?"

# --- Logger Setup ---
# Configure logger for Streamlit (optional, but good for debugging)
logger.remove()
logger.add(sys.stderr, level="DEBUG")  # You can change this to INFO for less verbosity

# --- Helper function to encode images for HTML embedding ---
def get_image_as_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- System Initialization ---
@st.cache_resource  # Cache the system resource to avoid reloading on every interaction
def load_collaboration_system():
    """Loads and initializes the CollaborationSystem."""
    try:
        init_api(read_json(API_CONFIG_PATH))
        # web_demo=True enables system.web_log and system.log() to use st.markdown
        system = CollaborationSystem(config_path=SYSTEM_CONFIG_PATH, task='chat', web_demo=True)
        logger.info("CollaborationSystem loaded successfully.")
        return system
    except FileNotFoundError as e:
        st.error(
            f"Configuration file not found: {e}. Please ensure '{API_CONFIG_PATH}' and '{SYSTEM_CONFIG_PATH}' exist.")
        logger.error(f"Error loading system: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while initializing the system: {e}")
        logger.error(f"Error loading system: {e}")
        return None


def display_chat_messages():
    """Displays chat messages from session state."""
    for message in st.session_state.messages:
        avatar = None
        if message["role"] == "user":
            avatar = "👤"
        elif message["role"] == "assistant":
            avatar = get_avatar("Manager")  # Use Manager's avatar for final response

        with st.chat_message(message["role"], avatar=avatar):
            # Check if the content is a list (internal logs + final response structure)
            if isinstance(message["content"], list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict) and "type" in content_item:
                        if content_item["type"] == "internal_logs_container":
                            if st.session_state.dev_mode:  # Only show if dev_mode is True
                                with st.expander("🔍 Agent Interactions", expanded=False):
                                    for log_entry in content_item["logs"]:
                                        # Assuming log_entry is already formatted markdown from system.web_log
                                        st.markdown(log_entry, unsafe_allow_html=True)
                        elif content_item["type"] == "final_response":
                            # Highlight the final response
                            st.markdown(f"""
                            <div style="background-color: #e6f7ff; border-left: 5px solid #1890ff; padding: 10px; border-radius: 5px; margin-bottom:10px;">
                                {content_item["response"]}
                            </div>
                            """, unsafe_allow_html=True)
                    elif isinstance(content_item, str):  # Simple string message
                        st.markdown(content_item, unsafe_allow_html=True)

            elif isinstance(message["content"], str):  # Simple string message (e.g. initial greeting)
                st.markdown(message["content"], unsafe_allow_html=True)


def main_chat_interface():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🛋️", layout="wide")

    # --- Header with 3 Columns: [Logo1 Logo2 (centered)] [Title] [Logo3] ---
    col1_logos_container, col2_title, col3_logo = st.columns([2, 6, 1])  # Adjust ratios as needed

    with col1_logos_container:
        # Get image data as base64
        logo1_b64 = get_image_as_base64(LOGO_PATH_1)
        logo2_b64 = get_image_as_base64(LOGO_PATH_2)
        logo_width = 80  # Define a common width for the logos in pixels

        html_logos_centered = f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            """
        if logo1_b64:
            html_logos_centered += f'<img src="data:image/png;base64,{logo1_b64}" width="{logo_width}" style="margin-right: 10px;">'
        else:
            html_logos_centered += '<span style="font-size: 30px; margin-right: 10px;">🛋️¹</span>'  # Placeholder

        if logo2_b64:
            html_logos_centered += f'<img src="data:image/png;base64,{logo2_b64}" width="{logo_width}">'
        else:
            html_logos_centered += '<span style="font-size: 30px;">🛋️²</span>'  # Placeholder

        html_logos_centered += "</div>"
        st.markdown(html_logos_centered, unsafe_allow_html=True)

    with col2_title:
        st.title(PAGE_TITLE)
        st.caption("Powered by SofAgent Collaborative System")

    with col3_logo:
        if os.path.exists(LOGO_PATH_3):
            st.image(LOGO_PATH_3, width=100)  # Adjust width as needed for this logo
        else:
            st.markdown("🛋️³", unsafe_allow_html=True)  # Use markdown for emoji for consistency

    st.markdown("---")

    # --- Sidebar for Mode Selection ---
    st.sidebar.title("Settings")
    dev_mode = st.sidebar.checkbox("🛠️ Development Mode (Show Agent Interactions)", value=True)
    if 'dev_mode' not in st.session_state:
        st.session_state.dev_mode = dev_mode
    elif st.session_state.dev_mode != dev_mode:
        st.session_state.dev_mode = dev_mode
        # No rerun needed here, will be picked up on next message display

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": INITIAL_GREETING}]
        if 'system' in st.session_state and st.session_state.system is not None:
            st.session_state.system.reset(clear_session_history=True)  # Reset system state as well
        st.rerun()

    # --- Load System ---
    if 'system' not in st.session_state:
        st.session_state.system = None  # Initialize if not present

    if st.session_state.system is None:  # Attempt to load if not already loaded
        # Wrap in a button to allow user to initiate loading if it fails once
        if st.button("Initialize SofAgent System"):
            with st.spinner("Loading SofAgent System... Please wait."):
                st.session_state.system = load_collaboration_system()
            if st.session_state.system:
                st.success("System initialized successfully!")
                st.rerun()  # Rerun to reflect the loaded state
            else:
                st.error("System failed to initialize. Please check logs and configuration.")
        return  # Stop further execution if system is not loaded

    system = st.session_state.system
    if system is None:  # If loading failed or wasn't triggered.
        st.warning("SofAgent system is not loaded. Click 'Initialize SofAgent System' to start.")
        return

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": INITIAL_GREETING}]

    # --- Display Existing Chat Messages ---
    display_chat_messages()

    # --- User Input ---
    if prompt := st.chat_input("Chiedi a SofAgent informazioni su divani, arredi o abbinamenti..."):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Prepare container for assistant's response
        assistant_response_parts = []

        # Display "thinking" animation and process the request
        with st.chat_message("assistant", avatar=get_avatar("Manager")):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🧠 SofAgent is thinking...")

            try:
                # Call the system. system.web_log will be populated.
                # The system's log method might print live if web_demo=True.
                system.clear_web_log()  # Clear previous turn's web_log
                final_response_from_manager = system(prompt,
                                                     reset_session_history=False)  # reset=False to maintain chat history within system

                # Capture internal logs if in development mode
                internal_logs_for_display = []
                if st.session_state.dev_mode and hasattr(system, 'web_log'):
                    # Filter out redundant "Manager is thinking/acting" if needed, or just display all
                    # The system.log already formats with avatars.
                    internal_logs_for_display = list(system.web_log)  # Make a copy

                thinking_placeholder.empty()  # Clear "thinking" message

                # Structure the assistant's message content
                # 1. Container for internal logs (conditionally displayed by display_chat_messages)
                assistant_response_parts.append({
                    "type": "internal_logs_container",
                    "logs": internal_logs_for_display
                })
                # 2. The final, highlighted response
                assistant_response_parts.append({
                    "type": "final_response",
                    "response": final_response_from_manager
                })

            except Exception as e:
                thinking_placeholder.empty()
                logger.error(f"Error during system call: {e}", exc_info=True)
                st.error(f"An error occurred: {e}")
                # Add error as assistant message
                assistant_response_parts.append(f"Sorry, I encountered an error: {e}")

        # Add the structured assistant response to session_state.messages
        st.session_state.messages.append({"role": "assistant", "content": assistant_response_parts})
        st.rerun()  # Rerun to display the new messages


if __name__ == "__main__":
    # This allows running this page directly: streamlit run sofagent/pages/sofagent_ui.py
    main_chat_interface()