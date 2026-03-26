#!/usr/bin/env python3
"""
OpenClaw Control - A clean, modern GUI for managing agents and chatting
"""

import streamlit as st
import subprocess
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE_DIR = Path.home() / ".openclaw" / "workspace-developer"
AVATARS_DIR = WORKSPACE_DIR / "avatars"
AVATARS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="OpenClaw Control",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.openclaw.ai',
        'Report a bug': None,
        'About': 'OpenClaw Control - Agent Management Interface'
    }
)

# Modern dark theme CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background: #0d0d0d;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1a1a1a;
        border-right: 1px solid #2d2d2d;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: #10a37f;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Agent cards in sidebar */
    .agent-card {
        background: #262626;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .agent-card:hover {
        background: #333;
        border-color: #10a37f;
    }
    
    .agent-card.selected {
        background: #10a37f20;
        border-color: #10a37f;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: transparent !important;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
        background: #262626;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #333;
    }
    
    /* User messages */
    .stChatMessage[data-testid="stChatMessage"][data-role="user"] [data-testid="stChatMessageContent"] {
        background: #10a37f20;
        border-color: #10a37f40;
    }
    
    /* Input box */
    .stChatInputContainer {
        background: #1a1a1a;
        border-top: 1px solid #333;
    }
    
    /* Buttons */
    .stButton > button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #0d8a6a;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #333;
        color: #e0e0e0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #262626;
        border-radius: 6px;
        border: 1px solid #333;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #262626;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 6px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: #262626;
        border: 2px dashed #444;
        border-radius: 8px;
    }
    
    /* Welcome screen */
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def run_openclaw_command(args: list) -> tuple:
    """Run an OpenClaw CLI command and return (stdout, stderr, returncode)"""
    try:
        result = subprocess.run(
            ["openclaw"] + args,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except FileNotFoundError:
        return "", "OpenClaw CLI not found. Make sure it's installed and in PATH.", 1
    except Exception as e:
        return "", str(e), 1


def get_agents_list():
    """Get list of available agents from OpenClaw CLI"""
    stdout, stderr, rc = run_openclaw_command(["agents", "list"])
    if rc != 0:
        return []
    
    agents = []
    for line in stdout.strip().split('\n'):
        line = line.strip()
        # Skip empty lines, separators, and UI characters
        if line and not line.startswith('-') and line not in ['│', '─', '├', '└'] and not all(c in '─│├└┌┐┤┴┬┼' for c in line):
            # Clean up any box-drawing characters
            clean_line = re.sub(r'[│─├└┌┐┤┴┬┼]', '', line).strip()
            if clean_line and clean_line not in agents:
                agents.append(clean_line)
    return sorted(agents)


def get_agent_avatar_path(agent_id: str) -> Path:
    """Get path to agent avatar image"""
    if not agent_id:
        return None
    for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        path = AVATARS_DIR / f"{agent_id}{ext}"
        if path.exists():
            return path
    return None


def render_agent_card(agent_id, is_selected=False):
    """Render a single agent card"""
    avatar_path = get_agent_avatar_path(agent_id)
    
    # Card styling
    border_color = "#10a37f" if is_selected else "#333"
    bg_color = "#10a37f20" if is_selected else "#262626"
    
    card_html = f"""
    <div style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.2s;
    " onmouseover="this.style.borderColor='#10a37f'" onmouseout="this.style.borderColor='{border_color}'">
        <div style="
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            overflow: hidden;
        ">
            {"<img src='" + str(avatar_path) + "' style='width:100%;height:100%;object-fit:cover;' />" if avatar_path else "🤖"}
        </div>
        <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 500; color: #e0e0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {agent_id}
            </div>
            <div style="font-size: 0.8rem; color: #888; margin-top: 2px;">
                Ready to chat
            </div>
        </div>
    </div>
    """
    return card_html


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "agents_cache" not in st.session_state:
    st.session_state.agents_cache = None
if "show_create" not in st.session_state:
    st.session_state.show_create = False


# SIDEBAR
with st.sidebar:
    # Header
    st.markdown("<h1 style='margin:0;'>🦖 OpenClaw</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888; margin:0 0 1rem 0;'>Agent Control Center</p>", unsafe_allow_html=True)
    
    # New Chat button (prominent)
    if st.button("➕ New Agent", use_container_width=True, type="primary"):
        st.session_state.show_create = True
        st.session_state.current_agent = None
        st.rerun()
    
    st.divider()
    
    # Refresh agents
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<p style='color:#888; font-size:0.9rem; margin:0;'>Your Agents</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄", key="refresh_btn"):
            st.session_state.agents_cache = None
            st.rerun()
    
    # Get agents
    if st.session_state.agents_cache is None:
        st.session_state.agents_cache = get_agents_list()
    
    agents = st.session_state.agents_cache
    
    # Agent list
    if not agents:
        st.info("No agents yet. Create your first one!", icon="💡")
    else:
        for agent in agents:
            is_selected = st.session_state.current_agent == agent
            
            # Use columns for layout instead of HTML
            cols = st.columns([1, 6])
            with cols[0]:
                avatar_path = get_agent_avatar_path(agent)
                if avatar_path:
                    st.image(str(avatar_path), width=40)
                else:
                    st.markdown("🤖")
            with cols[1]:
                btn_type = "primary" if is_selected else "secondary"
                if st.button(
                    agent,
                    key=f"btn_{hash(agent)}",
                    use_container_width=True,
                    type=btn_type
                ):
                    st.session_state.current_agent = agent
                    st.session_state.show_create = False
                    if agent not in st.session_state.messages:
                        st.session_state.messages[agent] = []
                    st.rerun()
    
    st.divider()
    
    # Settings / Config
    with st.expander("⚙️ Settings"):
        st.markdown("<p style='color:#888; font-size:0.8rem;'>Workspace</p>", unsafe_allow_html=True)
        st.code(str(WORKSPACE_DIR), language=None)
        
        st.markdown("<p style='color:#888; font-size:0.8rem; margin-top:1rem;'>Avatars</p>", unsafe_allow_html=True)
        avatar_count = len(list(AVATARS_DIR.glob("*"))) if AVATARS_DIR.exists() else 0
        st.markdown(f"<p style='color:#10a37f;'>{avatar_count} avatars</p>", unsafe_allow_html=True)


# MAIN CONTENT AREA
if st.session_state.show_create:
    # CREATE AGENT VIEW
    st.markdown("<h1 style='margin-bottom:2rem;'>Create New Agent</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_agent_id = st.text_input(
            "Agent Name",
            placeholder="e.g., coding-assistant, research-bot",
            help="A unique identifier for your agent"
        )
        
        new_agent_model = st.selectbox(
            "Model",
            [
                "gpt-4o",
                "gpt-4o-mini",
                "claude-sonnet-4",
                "claude-opus-4",
                "ollama/llama3.2",
                "ollama/mistral",
                "ollama/codellama",
                "ollama/kimi-k2.5"
            ],
            help="Choose the AI model for this agent"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        btn_cols = st.columns([1, 1])
        with btn_cols[0]:
            if st.button("Create Agent", use_container_width=True, type="primary"):
                if new_agent_id:
                    with st.spinner("Creating agent..."):
                        stdout, stderr, rc = run_openclaw_command([
                            "agents", "add", new_agent_id,
                            "--model", new_agent_model
                        ])
                        if rc == 0:
                            st.success(f"✅ Created '{new_agent_id}'")
                            st.session_state.agents_cache = None
                            time.sleep(1)
                            st.session_state.show_create = False
                            st.session_state.current_agent = new_agent_id
                            st.rerun()
                        else:
                            st.error(f"Failed: {stderr or stdout}")
                else:
                    st.warning("Please enter an agent name")
        
        with btn_cols[1]:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_create = False
                st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:1rem;">
            <h4 style="margin-top:0;">💡 Tips</h4>
            <ul style="color:#888; padding-left:1.2rem;">
                <li>Use descriptive names</li>
                <li>GPT-4o is best for general tasks</li>
                <li>Claude excels at coding</li>
                <li>Ollama models run locally</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


elif st.session_state.current_agent is None:
    # WELCOME SCREEN
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; max-width: 700px; margin: 0 auto;">
        <div style="font-size: 3rem; font-weight: 700; margin-bottom: 1rem; background: linear-gradient(135deg, #10a37f, #4ade80); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Welcome to OpenClaw
        </div>
        <div style="font-size: 1.1rem; color: #888; margin-bottom: 2rem;">
            Your personal AI agent control center. Create agents, customize them with avatars, 
            and start conversations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards using Streamlit columns
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #1a1a1a; border: 1px solid #2d2d2d; border-radius: 8px; padding: 1.5rem; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
            <div style="font-weight: 600; color: #e0e0e0; margin-bottom: 0.5rem;">Multiple Agents</div>
            <div style="color: #888; font-size: 0.9rem;">Create specialized agents for different tasks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #1a1a1a; border: 1px solid #2d2d2d; border-radius: 8px; padding: 1.5rem; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎨</div>
            <div style="font-weight: 600; color: #e0e0e0; margin-bottom: 0.5rem;">Custom Avatars</div>
            <div style="color: #888; font-size: 0.9rem;">Personalize each agent with unique images</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div style="background: #1a1a1a; border: 1px solid #2d2d2d; border-radius: 8px; padding: 1.5rem; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
            <div style="font-weight: 600; color: #e0e0e0; margin-bottom: 0.5rem;">Fast Setup</div>
            <div style="color: #888; font-size: 0.9rem;">Get started in seconds with pre-configured models</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background: #1a1a1a; border: 1px solid #2d2d2d; border-radius: 8px; padding: 1.5rem; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
            <div style="font-weight: 600; color: #e0e0e0; margin-bottom: 0.5rem;">Persistent Chat</div>
            <div style="color: #888; font-size: 0.9rem;">Each agent remembers your conversation</div>
        </div>
        """, unsafe_allow_html=True)


else:
    # CHAT INTERFACE
    agent = st.session_state.current_agent
    avatar_path = get_agent_avatar_path(agent)
    
    # Header with agent info
    header_cols = st.columns([1, 10, 1])
    
    with header_cols[0]:
        if avatar_path:
            st.image(str(avatar_path), width=50)
        else:
            st.markdown("<div style='font-size:2rem; text-align:center;'>🤖</div>", unsafe_allow_html=True)
    
    with header_cols[1]:
        st.markdown(f"<h2 style='margin:0;'>{agent}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#888; margin:0;'>Online • Ready to chat</p>", unsafe_allow_html=True)
    
    with header_cols[2]:
        # Avatar upload button
        uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "gif", "webp"], key=f"avatar_{agent}", label_visibility="collapsed")
        if uploaded:
            # Save avatar
            ext = Path(uploaded.name).suffix.lower()
            save_path = AVATARS_DIR / f"{agent}{ext}"
            with open(save_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.toast(f"Avatar updated for {agent}!")
            st.rerun()
    
    st.divider()
    
    # Initialize chat history
    if agent not in st.session_state.messages:
        st.session_state.messages[agent] = []
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        # Welcome message if empty
        if not st.session_state.messages[agent]:
            st.markdown(f"""
            <div style="text-align:center; padding:3rem; color:#666;">
                <div style="font-size:3rem; margin-bottom:1rem;">👋</div>
                <div style="font-size:1.2rem; margin-bottom:0.5rem;">Start chatting with {agent}</div>
                <div style="font-size:0.9rem;">Type a message below to begin</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for msg in st.session_state.messages[agent]:
            with st.chat_message(msg["role"], avatar=msg.get("avatar") if msg.get("avatar") else ("🧑" if msg["role"] == "user" else "🤖")):
                st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Message...", key=f"chat_input_{agent}"):
        # Add user message
        user_msg = {"role": "user", "content": prompt, "avatar": "🧑"}
        st.session_state.messages[agent].append(user_msg)
        
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(""):
                stdout, stderr, rc = run_openclaw_command([
                    "sessions", "send",
                    "--agent", agent,
                    "--message", prompt,
                    "--timeout", "120"
                ])
                
                if rc == 0:
                    response = stdout.strip()
                else:
                    response = f"❌ Error: {stderr or 'Failed to get response'}"
                
                st.markdown(response)
        
        # Save assistant response
        assistant_msg = {
            "role": "assistant",
            "content": response,
            "avatar": str(avatar_path) if avatar_path else "🤖"
        }
        st.session_state.messages[agent].append(assistant_msg)


# Footer
st.divider()
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.8rem; padding:1rem 0;">
    OpenClaw Control v1.0 • Built with Streamlit
</div>
""", unsafe_allow_html=True)
