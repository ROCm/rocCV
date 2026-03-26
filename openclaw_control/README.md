# OpenClaw Control

A lightweight GUI for managing OpenClaw agents and chatting with them.

![OpenClaw Control](https://img.shields.io/badge/OpenClaw-Control-blue)

## Features

- 🤖 **Agent Management**: Create, list, and switch between agents
- 💬 **Persistent Chat**: Each agent has its own chat history
- 🎨 **Custom Avatars**: Upload images to personalize your agents
- ⚙️ **Model Selection**: Choose from multiple LLM providers
- 🔄 **Live Refresh**: Refresh agent list on-demand

## Installation

1. Make sure you have OpenClaw CLI installed and in your PATH
2. Install Python dependencies:

```bash
cd openclaw_control
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Quick Start

1. **Create an Agent**: Click "➕ Create Agent" in the sidebar, enter an ID and select a model
2. **Upload Avatar**: Click "🎨 Upload Avatar" to add a profile picture to your agent
3. **Start Chatting**: Click on an agent in the sidebar to open the chat interface
4. **Manage Multiple**: Switch between agents anytime - each has its own chat history

## Directory Structure

```
openclaw_control/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

Avatars are stored in `~/.openclaw/workspace-developer/avatars/`

## Requirements

- Python 3.8+
- OpenClaw CLI installed and configured
- Streamlit

## Troubleshooting

### "OpenClaw CLI not found"
Make sure `openclaw` is in your system PATH. Test with:
```bash
which openclaw
openclaw --version
```

### Agents not showing
Click the 🔄 refresh button in the sidebar to reload the agent list.

### Chat not working
Ensure the OpenClaw gateway is running:
```bash
openclaw gateway status
```

## License

MIT
