#!/usr/bin/env python3
"""
OpenClaw Control - HTML/JS Web Interface
Flask backend with modern frontend
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import json
import os
import re
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
WORKSPACE_DIR = Path.home() / ".openclaw" / "workspace-developer"
AVATARS_DIR = WORKSPACE_DIR / "avatars"
AVATARS_DIR.mkdir(exist_ok=True)

# In-memory storage for chat history (per agent)
chat_history = {}


def run_openclaw_command(args: list) -> tuple:
    """Run an OpenClaw CLI command"""
    try:
        result = subprocess.run(
            ["openclaw"] + args,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except FileNotFoundError:
        return "", "OpenClaw CLI not found", 1
    except Exception as e:
        return "", str(e), 1


def get_agents_list():
    """Get list of available agents with their identity information"""
    stdout, stderr, rc = run_openclaw_command(["agents", "list"])
    if rc != 0:
        return []

    agents = []
    current_agent = None

    for line in stdout.strip().split('\n'):
        line = line.strip()

        # Match agent header line like "- main (default) (Manager)"
        agent_match = re.match(r'^[-*]\s+(\S+)\s*\(.*\)', line)
        if agent_match:
            agent_id = agent_match.group(1)
            current_agent = {"id": agent_id, "name": agent_id}
            agents.append(current_agent)
            continue

        # Match identity line like "Identity: Simon (IDENTITY.md)"
        identity_match = re.search(r'Identity:\s+(.+?)(?:\s*\(|$)', line)
        if identity_match and current_agent:
            identity_name = identity_match.group(1).strip()
            # Remove emoji from the beginning if present and store separately
            emoji_match = re.match(r'^(\S+)\s+(.+)$', identity_name)
            if emoji_match and len(emoji_match.group(1)) <= 2:
                current_agent["emoji"] = emoji_match.group(1)
                current_agent["name"] = emoji_match.group(2).strip()
            else:
                current_agent["name"] = identity_name
            continue

    return agents


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'enhanced_ui.html')


@app.route('/classic')
def classic_ui():
    """Serve the classic HTML page"""
    return send_from_directory('.', 'index.html')


@app.route('/api/agents', methods=['GET'])
def list_agents():
    """List all agents with their identity information"""
    agents = get_agents_list()
    return jsonify({"agents": agents})


@app.route('/api/agents', methods=['POST'])
def create_agent():
    """Create a new agent"""
    data = request.get_json()
    agent_id = data.get('id')
    model = data.get('model', 'gpt-4o')
    
    if not agent_id:
        return jsonify({"error": "Agent ID required"}), 400
    
    stdout, stderr, rc = run_openclaw_command([
        "agents", "add", agent_id,
        "--model", model
    ])
    
    if rc == 0:
        return jsonify({"success": True, "id": agent_id})
    else:
        return jsonify({"error": stderr or stdout}), 500


@app.route('/api/agents/<agent_id>/avatar', methods=['POST'])
def upload_avatar(agent_id):
    """Upload avatar for an agent"""
    if 'avatar' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save avatar
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        return jsonify({"error": "Invalid file type"}), 400
    
    save_path = AVATARS_DIR / f"{agent_id}{ext}"
    file.save(save_path)
    
    return jsonify({"success": True, "path": str(save_path)})


@app.route('/api/agents/<agent_id>/avatar', methods=['GET'])
def get_avatar(agent_id):
    """Get avatar for an agent"""
    for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        path = AVATARS_DIR / f"{agent_id}{ext}"
        if path.exists():
            return send_from_directory(AVATARS_DIR, f"{agent_id}{ext}")
    return jsonify({"error": "No avatar found"}), 404


@app.route('/api/chat/<agent_id>', methods=['GET'])
def get_chat_history(agent_id):
    """Get chat history for an agent"""
    return jsonify({"messages": chat_history.get(agent_id, [])})


@app.route('/api/chat/<agent_id>', methods=['POST'])
def send_message(agent_id):
    """Send a message to an agent"""
    data = request.get_json()
    message = data.get('message')
    
    if not message:
        return jsonify({"error": "Message required"}), 400
    
    # Store user message
    if agent_id not in chat_history:
        chat_history[agent_id] = []
    
    chat_history[agent_id].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get response from OpenClaw
    stdout, stderr, rc = run_openclaw_command([
        "sessions", "send",
        "--agent", agent_id,
        "--message", message,
        "--timeout", "120"
    ])
    
    if rc == 0:
        response = stdout.strip()
    else:
        response = f"Error: {stderr or 'Failed to get response'}"
    
    # Store assistant response
    chat_history[agent_id].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })
    
    return jsonify({"response": response})


@app.route('/api/chat/<agent_id>', methods=['DELETE'])
def clear_chat(agent_id):
    """Clear chat history for an agent"""
    if agent_id in chat_history:
        chat_history[agent_id] = []
    return jsonify({"success": True})


if __name__ == '__main__':
    print("🦖 OpenClaw Control")
    print("=" * 40)
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, debug=True)
