import socket
import time
import threading
import queue
import numpy as np
import soundfile as sf
import os
import sys
import base64
import json
import paho.mqtt.client as mqtt
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, redirect, url_for
from werkzeug.serving import make_server
import requests
import uuid
from flask import render_template
import datetime
from collections import deque # Added import
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Configuration ---
TRANSCRIPTION_LANGUAGE = os.getenv('TRANSCRIPTION_LANGUAGE', 'en')
UDP_HOST = os.getenv('UDP_HOST', '0.0.0.0')
UDP_PORT = int(os.getenv('UDP_PORT', '5005'))
IMAGE_UDP_PORT = int(os.getenv('IMAGE_UDP_PORT', '5007'))
HTTP_HOST = os.getenv('HTTP_HOST', '0.0.0.0')
HTTP_PORT = int(os.getenv('PORT', '5006'))
SPEECH_UDP_SERVER_UDP_HOST = '134.199.220.52'
SPEECH_UDP_SERVER_UDP_PORT = 5005
SPEECH_UDP_SERVER_IMAGE_UDP_PORT = 5007
SPEECH_UDP_SERVER_HTTP_HOST = '134.199.220.52'
SPEECH_UDP_SERVER_HTTP_PORT = 5006
BUFFER_SIZE = 65536
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
AUDIO_CHUNKS_DIR = "audio_chunks"
OUTPUT_AUDIO_DIR = "output_audio"
RECEIVED_IMAGES_DIR = "received_images"

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', '134.199.220.52')  # Changed from localhost
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))  # Changed from 6000
MQTT_CLIENT_ID_SERVER = f"cheeko_server_{os.getpid()}"

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE = os.getenv('ELEVENLABS_VOICE', "Laura")
ELEVENLABS_VOICE_MAP = {
    "sparkles for kids": "SmbeE2y2kj7VTBEUvnAw",
    "deep voice": "CZdRaSQ51p0onta4eec8",
    "soft calm voice": "zgqefOY5FPQ3bB7OZTVR",
}
DEFAULT_ELEVENLABS_VOICE = "Laura"
# --- Logging Setup ---
log_messages = deque(maxlen=100) # Store last 100 log messages

def log_message(message):
    """Prints a message and adds it to the log deque."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, file=sys.stderr) # Keep printing to stderr for console visibility
    log_messages.append(log_entry)

# --- Global Variables ---
audio_buffer = {}
image_buffer = {}
pending_images = {}
pending_images_lock = threading.Lock()
last_transcription_time = {}
transcription_lock = threading.Lock()
running = True
groq_client = None
conversation_history = {}
MAX_HISTORY_MESSAGES = 8
mqtt_client_server = None
http_server_thread = None
client_tokens = {} # Stores active client info {client_id: {token, expires_at, role, voice, language_code, language}}

# --- Toy Personas ---
TOY_PERSONAS = {
    "Story Teller": (
        "You are Cheeko, a friendly and playful toy that answers kids' questions in English. "
        "You respond in a curious, storytelling manner with fun and simple explanations under 100 words. "
        "You only share safe, age-appropriate information and never discuss sensitive, scary, or inappropriate topics."
    ),
    "Puzzle Solver": (
        "You are a clever puzzle solver. You help kids solve riddles, puzzles, and brain teasers. "
        "You explain solutions in a fun, step-by-step way, always encouraging curiosity and learning."
    ),
    "Math Tutor": (
        "You are a friendly math tutor. You help kids understand math concepts and solve math problems. "
        "You explain things simply, use examples, and always keep explanations under 100 words."
    ),
}
DEFAULT_PERSONA = TOY_PERSONAS["Story Teller"]

LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr"
}

# --- Image Analysis Functions ---
def encode_image(image_path):
    """Encodes the image as base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        log_message(f"[Image Encode Error] Failed to encode image {image_path}: {e}")
        raise

def analyze_image_with_llama(image_path, client_id, transcription=""):
    """Analyzes the image using meta-llama/llama-4-scout-17b-16e-instruct on Groq, incorporating user transcription and language."""
    global conversation_history
    try:
        base64_image = encode_image(image_path)
        # Get user's preferred language
        language_name = client_tokens.get(client_id, {}).get("language", "English")
        language_instruction = f"Always reply in {language_name}.\n"
        base_prompt = (
            "You are Cheeko, a friendly toy talking to a young child. Describe what's in this image in a fun, simple way,  "
            "as if you're explaining it to a 5-year-old. Use short sentences and avoid complex words. always keep replay under 50 words, "
        )
        if transcription:
            prompt = (
                f"{language_instruction}{base_prompt} The child asked: '{transcription}'. Answer their question about the image in your response, "
                "keeping the explanation fun, simple, and appropriate for a young child."
            )
        else:
            prompt = language_instruction + base_prompt
        log_message(f"[Image Analysis] Sending image for analysis with final prompt: {prompt}")
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=500
        )
        analysis_result = response.choices[0].message.content.strip()
        log_message(f"[Image Analysis] LLaMA Scout analysis for {image_path} with transcription '{transcription}'")

        # --- Add to conversation history like get_toy_response ---
        if client_id not in conversation_history:
            conversation_history[client_id] = []
        if transcription:
            conversation_history[client_id].append({"role": "user", "content": transcription})
        conversation_history[client_id].append({"role": "assistant", "content": analysis_result})
        if len(conversation_history[client_id]) > MAX_HISTORY_MESSAGES:
            conversation_history[client_id] = conversation_history[client_id][-MAX_HISTORY_MESSAGES:]
        # --------------------------------------------------------

        return analysis_result
    except Exception as e:
        log_message(f"[Image Analysis Error] Failed to analyze image {image_path}: {e}")
        return f"Sorry, I couldn't look at the picture clearly! Can you tell me more?"

# --- Image Handling Functions ---
def save_received_image(image_data, client_id, image_id):
    """Stores the received image and sends imagereceived MQTT notification."""
    try:
        os.makedirs(RECEIVED_IMAGES_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_path = os.path.join(RECEIVED_IMAGES_DIR, f"image_{client_id}_{image_id}_{timestamp}.jpeg")
        
        # Decode base64 data
        try:
            image_bytes = base64.b64decode(image_data)
        except base64.binascii.Error as e:
            log_message(f"[Image Error] Invalid base64 data for client {client_id}, image ID {image_id}: {e}")
            notify_image_status(client_id, os.path.basename(output_path), "imagereceived", f"Invalid base64 data: {e}")
            return
        
        # Save the image
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        log_message(f"[Image] Saved received image to: {output_path}")

        # Store image path for later analysis
        with pending_images_lock:
            pending_images[client_id] = output_path
        log_message(f"[Image] Stored image path for client {client_id}: {output_path}")

        # Send imagereceived notification
        notify_image_status(client_id, output_path, "imagereceived", None)
    except Exception as e:
        log_message(f"[Image Error] Failed to save image for client {client_id}, image ID {image_id}: {e}")
        notify_image_status(client_id, f"failed_image_{image_id}", "imagereceived", f"Error: {str(e)}")


def notify_image_status(client_id, image_path, identifier, analysis_result):
    """Sends an MQTT notification to the client about image receipt or analysis."""
    try:
        msg_id = int(time.time() * 1000) % 1000000
        notification_payload = {
            "msgId": msg_id,
            "identifier": identifier,
            "inputParams": {
                "filename": os.path.basename(image_path),
                "received_at": int(time.time())
            }
        }
        
        # Add audio URL only for imagereceived notifications
        if identifier == "imagereceived":
            notification_payload["inputParams"]["url"] = f"http://{SPEECH_UDP_SERVER_HTTP_HOST}:{SPEECH_UDP_SERVER_HTTP_PORT}/default_audio/imagereceived.mp3"
            
        # Add analysis result for imageanalyzed notifications
        if identifier == "imageanalyzed" and analysis_result is not None:
            notification_payload["inputParams"]["analysis"] = analysis_result
            
        target_topic = f"user/cheekotoy/{client_id}/thing/command/call"
        if mqtt_client_server and mqtt_client_server.is_connected():
            payload_str = json.dumps(notification_payload, ensure_ascii=False)
            result = mqtt_client_server.publish(target_topic, payload_str.encode('utf-8'), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log_message(f"[MQTT Server] Notified client {client_id} with {identifier}: {image_path}")
            else:
                log_message(f"[MQTT Server] Failed to notify {identifier}. RC: {result.rc}")
        else:
            log_message(f"[MQTT Server] Cannot notify {identifier}, client not connected.")
    except Exception as e:
        log_message(f"[MQTT Error] Failed to send {identifier} notification: {e}")

# --- MQTT Callbacks ---
def on_connect_server(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log_message("[MQTT Server] Connected to MQTT Broker!")
        client.subscribe("user/cheekotoy/+/thing/command/callAck")
        client.subscribe("user/cheekotoy/+/thing/data/post")
        client.subscribe("user/cheekotoy/+/thing/event/post")
        client.subscribe("config")
        log_message("[MQTT Server] Subscribed to topics: user/cheekotoy/+/thing/command/callAck, user/cheekotoy/+/thing/data/post, user/cheekotoy/+/thing/event/post, config")
    else:
        log_message(f"[MQTT Server] Failed to connect, return code {rc}")

def handle_config_request(data):
    try:
        if "client_id" not in data:
            log_message("[MQTT Warning] Received config request missing 'client_id'")
            return
        client_id = data["client_id"]
        msg_id = data.get("msgId", int(time.time() * 1000))
        token = data.get("token")
        if token and is_valid_token(token, client_id):
            log_message(f"[Config Request] Valid token for client: {client_id}")
        else:
            log_message(f"[Config Request] Invalid or no token for client: {client_id}")
            return
        config_payload = {
            "msgId": msg_id,
            "identifier": "updateconfig",
            "inputParams": {
                "udp_host": SPEECH_UDP_SERVER_UDP_HOST,
                "udp_port": SPEECH_UDP_SERVER_UDP_PORT,
                "image_udp_port": SPEECH_UDP_SERVER_IMAGE_UDP_PORT,
                "http_port": SPEECH_UDP_SERVER_HTTP_PORT,
                "http_host": SPEECH_UDP_SERVER_HTTP_HOST,
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "updated_at": int(time.time())
            }
        }
        target_topic = f"user/cheekotoy/{client_id}/thing/command/call"
        if mqtt_client_server and mqtt_client_server.is_connected():
            payload_str = json.dumps(config_payload)
            result = mqtt_client_server.publish(target_topic, payload=payload_str, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log_message(f"[Config Request] Successfully sent config to {target_topic}")
            else:
                log_message(f"[Config Request] Failed to send config to {target_topic}")
        else:
            log_message("[MQTT Server] Cannot publish config, client not connected.")
    except Exception as e:
        log_message(f"[Config Request Error] Failed to process config request: {e}")

def on_message_server(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode("utf-8")
    log_message(f"[MQTT Server] Received message on `{topic}`: {payload}") # Log received MQTT messages
    try:
        data = json.loads(payload)
        topic_parts = topic.split('/')
        if len(topic_parts) >= 4:
            client_id = topic_parts[2]
        else:
            client_id = None
        if "thing/event/post" in topic and data.get("identifier") == "login":
            log_message(f"[Login] Received login event from client {client_id}: {data}")
            if not is_valid_toy(client_id):
                log_message(f"[Login] Toy with client_id {client_id} not found in Supabase. Rejecting login.")
                return
            role_type = get_toy_role_type(client_id)
            voice_type = get_toy_voice(client_id)
            language_name = get_toy_language(client_id) or "English"
            language_code = LANGUAGE_MAP.get(language_name, "en")
            log_message(f"[Login] Retrieved role_type '{role_type}', voice '{voice_type}', language_code '{language_code}' for client {client_id}")
            if not role_type:
                log_message(f"[Login] No role_type found for client {client_id}. Rejecting login.")
                return
            token = str(uuid.uuid4())
            expires_at = int(time.time()) + 3600
            client_tokens[client_id] = {
                "token": token,
                "expires_at": expires_at,
                "role": role_type,
                "voice": voice_type,
                "language_code": language_code,
                "language": language_name
            }
            log_message(f"[Login] Stored token, role_type '{role_type}', voice '{voice_type}', language_code '{language_code}' and language '{language_name}' for client {client_id}")
            token_response = {
                "msgId": int(time.time() * 1000),
                "identifier": "updatetoken",
                "inputParams": {
                    "token": token,
                    "expires_at": expires_at
                }
            }
            response_topic = f"user/cheekotoy/{client_id}/thing/command/call"
            result = mqtt_client_server.publish(response_topic, json.dumps(token_response), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log_message(f"[Login] Successfully sent token to client {client_id} on {response_topic}: {token[:8]}...")
            else:
                log_message(f"[Login] Failed to send token to client {client_id}. RC: {result.rc}")
            return
        elif topic == "config":
            log_message(f"[Config] Received config request: {data}")
            handle_config_request(data)
            return
        elif "thing/command/callAck" in topic:
            log_message(f"[Ack] Received acknowledgment from {client_id}: {data}")
            identifier = data.get("identifier")
            msg_id = data.get("msgId")
            result = data.get("result", 0)
            if identifier == "updatetoken" and result == 1:
                log_message(f"[Ack] Client {client_id} confirmed successful token update (msgId: {msg_id})")
                config_payload = {
                    "msgId": int(time.time() * 1000),
                    "identifier": "updateconfig",
                    "inputParams": {
                        "udp_host": SPEECH_UDP_SERVER_UDP_HOST,
                        "udp_port": SPEECH_UDP_SERVER_UDP_PORT,
                        "image_udp_port": SPEECH_UDP_SERVER_IMAGE_UDP_PORT,
                        "http_port": SPEECH_UDP_SERVER_HTTP_PORT,
                        "http_host": SPEECH_UDP_SERVER_HTTP_HOST,
                        "sample_rate": SAMPLE_RATE,
                        "channels": CHANNELS,
                        "updated_at": int(time.time())
                    }
                }
                config_topic = f"user/cheekotoy/{client_id}/thing/command/call"
                if mqtt_client_server and mqtt_client_server.is_connected():
                    payload_str = json.dumps(config_payload)
                    result = mqtt_client_server.publish(config_topic, payload_str, qos=1)
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        log_message(f"[Config] Successfully sent configuration to {config_topic}")
                    else:
                        log_message(f"[Config] Failed to send configuration to {config_topic}. RC: {result.rc}")
                else:
                    log_message("[MQTT Server] Cannot publish configuration, client not connected.")
            elif identifier == "audioplay":
                if result == 1:
                    log_message(f"[Ack] Client {client_id} confirmed playback of message {msg_id}")
                    cleanup_audio_files(client_id)
                else:
                    error_msg = data.get("error_msg", "Unknown error")
                    log_message(f"[Ack Error] Playback failed for {client_id}: {error_msg}")
            elif identifier in ["imagereceived", "imageanalyzed"]:
                if result == 1:
                    log_message(f"[Ack] Client {client_id} confirmed receipt of {identifier} notification (msgId: {msg_id})")
                else:
                    error_msg = data.get("error_msg", "Unknown error")
                    log_message(f"[Ack Error] {identifier} notification failed for {client_id}: {error_msg}")
            return
        elif "thing/data/post" in topic:
            log_message(f"[Data] Received data post from {client_id}: {data}")
            if data.get("identifier") == "updaterole":
                log_message(f"[Updaterole] Received updaterole event for client {client_id}")
                if client_id in client_tokens:
                    new_role = get_toy_role_type(client_id)
                    new_voice = get_toy_voice(client_id)
                    new_language_name = get_toy_language(client_id) or "English"
                    new_language_code = LANGUAGE_MAP.get(new_language_name, "en")
                    client_tokens[client_id].update({
                        "role": new_role,
                        "voice": new_voice,
                        "language_code": new_language_code,
                        "language": new_language_name
                    })
                    log_message(f"[Updaterole] Updated client_settings for {client_id}: role={new_role}, voice={new_voice}, language_code={new_language_code}, language={new_language_name}")
                else:
                    log_message(f"[Updaterole] Client {client_id} not found in client_tokens.")
            return
        log_message(f"[MQTT Server] Unhandled message on {topic}: {data}")
    except json.JSONDecodeError:
        log_message(f"[MQTT Server] Invalid JSON in payload: {payload}")
    except Exception as e:
        log_message(f"[MQTT Server Error] Processing message failed: {e}")

def is_valid_token(token, client_id):
    if client_id not in client_tokens:
        return False
    token_data = client_tokens.get(client_id)
    if not token_data:
        return False
    if token_data["token"] != token:
        return False
    if token_data["expires_at"] < int(time.time()):
        del client_tokens[client_id]
        return False
    return True


def generate_tts_audio2(text, filename, client_id):
    global mqtt_client_server
    log_message(f"[TTS] Generating audio using ElevenLabs for response: {filename}")
    try:
        # Initialize Supabase client with service role key
        supabase_url = "https://ozwvxxubkseztcakoobv.supabase.co"
        supabase_service_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96d3Z4eHVia3NlenRjYWtvb2J2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODcxNzc1OCwiZXhwIjoyMDU0MjkzNzU4fQ.7mdrn66Ma8dQiUdsTv8F5l4uEjqpd8gLi51hSevrEs8"
        supabase_anon_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96d3Z4eHVia3NlenRjYWtvb2J2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg3MTc3NTgsImV4cCI6MjA1NDI5Mzc1OH0.jVZtbMUn0fuQzprp-AP0Evz03v_6Zxq2s2RFEp8UlSk"
        if not supabase_service_key:
            raise Exception("SUPABASE_SERVICE_KEY not found in environment variables")
        if not supabase_anon_key:
            raise Exception("SUPABASE_ANON_KEY not found in environment variables")
        supabase: Client = create_client(supabase_url, supabase_service_key)

        # Generate audio with ElevenLabs
        voices_url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        response = requests.get(voices_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get voices: {response.text}")
        voices_data = response.json()["voices"]
        client_info = client_tokens.get(client_id, {})
        user_voice = client_info.get("voice", "")
        normalized_voice = user_voice.strip().lower()
        mapped_voice = ELEVENLABS_VOICE_MAP.get(normalized_voice, DEFAULT_ELEVENLABS_VOICE)
        selected_voice = next(
            (v for v in voices_data if v["voice_id"] == mapped_voice or v["name"].lower() == mapped_voice.lower()),
            None
        )
        if not selected_voice:
            log_message(f"[Warning] Voice {mapped_voice} not found, using first available voice")
            selected_voice = voices_data[0]
        log_message(f"[TTS] Using voice: {selected_voice['name']} ({selected_voice['voice_id']})")
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice['voice_id']}"
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        response = requests.post(tts_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to generate audio: {response.text}")
        audio_content = response.content

        # Try uploading to Supabase
        bucket_name = "toyresponse"
        supabase_path = f"audio/{client_id}/{filename}"
        public_url = None
        try:
            log_message(f"[Supabase] Uploading audio to bucket '{bucket_name}' at path '{supabase_path}'")
            supabase.storage.from_(bucket_name).upload(
                path=supabase_path,
                file=audio_content,
                file_options={"content-type": "audio/mpeg"}
            )
            # Generate public URL using anon key
            supabase_anon: Client = create_client(supabase_url, supabase_anon_key)
            public_url = supabase_anon.storage.from_(bucket_name).get_public_url(supabase_path)
            # Strip trailing '?' or query parameters
            if public_url.endswith('?'):
                public_url = public_url[:-1]
            
            log_message(f" {public_url}")
            log_message(f"[Supabase] Generated public URL: {public_url}")
        except Exception as supabase_error:
            log_message(f"[Supabase Error] Failed to upload to Supabase: {supabase_error}")
            if hasattr(supabase_error, 'response'):
                log_message(f"[Supabase Error Details] Response: {supabase_error.response.text}")
            # Fallback to local storage
            output_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
            os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(audio_content)
            # public_url = f"http://{SPEECH_UDP_SERVER_HTTP_HOST}:{SPEECH_UDP_SERVER_HTTP_PORT}/audio/{filename}"
            log_message(f"[TTS] Fallback: Saved audio locally and using local URL: {public_url}")

        # Send MQTT notification with the URL
        msg_id = int(time.time() * 1000) % 1000000
        notification_payload = {
            "msgId": msg_id,
            "identifier": "audioplay",
            "inputParams": {
                "recordingId": msg_id,
                "order": 1,
                "url": public_url
            }
        }
        target_topic = f"user/cheekotoy/{client_id}/thing/command/call"
        if mqtt_client_server and mqtt_client_server.is_connected():
            payload_str = json.dumps(notification_payload)
            result = mqtt_client_server.publish(target_topic, payload_str, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log_message(f"[MQTT Server] Published notification with URL for {filename} to {target_topic}")
            else:
                log_message(f"[MQTT Server] Failed to publish notification for {filename}. RC: {result.rc}")
        else:
            log_message("[MQTT Server] Cannot publish notification, client not connected.")

        # Save locally for debugging or dashboard
        if public_url.startswith("http://"):  # Already saved in fallback
            log_message(f"[TTS] Audio already saved locally to: {output_path}")
        else:
            output_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
            os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(audio_content)
            log_message(f"[TTS] Saved response audio locally to: {output_path}")

    except Exception as e:
        log_message(f"[Error] ElevenLabs TTS generation or processing failed: {e}")
        # Send default error audio
        msg_id = int(time.time() * 1000) % 1000000
        notification_payload = {
            "msgId": msg_id,
            "identifier": "audioplay",
            "inputParams": {
                "recordingId": msg_id,
                "order": 1,
                "url": f"http://{SPEECH_UDP_SERVER_HTTP_HOST}:{SPEECH_UDP_SERVER_HTTP_PORT}/default_audio/error.mp3"
            }
        }
        target_topic = f"user/cheekotoy/{client_id}/thing/command/call"
        if mqtt_client_server and mqtt_client_server.is_connected():
            payload_str = json.dumps(notification_payload)
            mqtt_client_server.publish(target_topic, payload_str, qos=1)

# old file
# def generate_tts_audio2(text, filename, client_id):
#     global mqtt_client_server
#     output_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
#     log_message(f"[TTS] Generating audio using ElevenLabs for response to: {output_path}")
#     try:
#         os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
#         voices_url = "https://api.elevenlabs.io/v1/voices"
#         headers = {"xi-api-key": ELEVENLABS_API_KEY}
#         response = requests.get(voices_url, headers=headers)
#         if response.status_code != 200:
#             raise Exception(f"Failed to get voices: {response.text}")
#         voices_data = response.json()["voices"]
#         client_info = client_tokens.get(client_id, {})
#         user_voice = client_info.get("voice", "")
#         normalized_voice = user_voice.strip().lower()
#         mapped_voice = ELEVENLABS_VOICE_MAP.get(normalized_voice, DEFAULT_ELEVENLABS_VOICE)
#         selected_voice = next(
#             (v for v in voices_data if v["voice_id"] == mapped_voice or v["name"].lower() == mapped_voice.lower()),
#             None
#         )
#         if not selected_voice:
#             log_message(f"[Warning] Voice {mapped_voice} not found, using first available voice")
#             selected_voice = voices_data[0]
#         log_message(f"[TTS] Using voice: {selected_voice['name']} ({selected_voice['voice_id']})")
#         tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice['voice_id']}"
#         payload = {
#             "text": text,
#             "model_id": "eleven_monolingual_v1",
#             "voice_settings": {
#                 "stability": 0.5,
#                 "similarity_boost": 0.75
#             }
#         }
#         response = requests.post(tts_url, headers=headers, json=payload)
#         if response.status_code != 200:
#             raise Exception(f"Failed to generate audio: {response.text}")
#         audio = response.content
#         with open(output_path, 'wb') as f:
#             f.write(audio)
#         log_message(f"[TTS] Saved response audio to: {output_path}")
#         audio_url = f"http://{SPEECH_UDP_SERVER_HTTP_HOST}:{SPEECH_UDP_SERVER_HTTP_PORT}/audio/{filename}"
#         msg_id = int(time.time() * 1000) % 1000000
#         notification_payload = {
#             "msgId": msg_id,
#             "identifier": "audioplay",
#             "inputParams": {
#                 "recordingId": msg_id,
#                 "order": 1,
#                 "url": audio_url
#             }
#         }
#         target_topic = f"user/cheekotoy/{client_id}/thing/command/call"
#         if mqtt_client_server and mqtt_client_server.is_connected():
#             payload_str = json.dumps(notification_payload)
#             result = mqtt_client_server.publish(target_topic, payload_str, qos=1)
#             if result.rc == mqtt.MQTT_ERR_SUCCESS:
#                 log_message(f"[MQTT Server] Published notification for {filename} to {target_topic}")
#             else:
#                 log_message(f"[MQTT Server] Failed to publish notification for {filename}. RC: {result.rc}")
#         else:
#             log_message("[MQTT Server] Cannot publish notification, client not connected.")
#     except Exception as e:
#         log_message(f"[Error] ElevenLabs TTS generation or MQTT publish failed: {e}")

def get_toy_response(transcript, client_id, role):
    global groq_client, conversation_history
    if not groq_client:
        log_message("[Error] Groq client not initialized.")
        return "Oops! I'm having trouble thinking right now."
    if client_id not in conversation_history:
        conversation_history[client_id] = []
    system_prompt = TOY_PERSONAS.get(role, DEFAULT_PERSONA)
    language_name = client_tokens[client_id].get("language", "English")
    system_prompt = f"keep replay under 50 words, Always reply in {language_name}.\n" + system_prompt
    log_message(f"[LLM] Using persona for role {role} for client {client_id}")
    log_message(f"[LLM] Sending transcript and history for client {client_id} to Groq...")
    try:
        messages_to_send = [{"role": "system", "content": system_prompt}]
        messages_to_send.extend(conversation_history[client_id])
        messages_to_send.append({"role": "user", "content": transcript})
        chat_completion = groq_client.chat.completions.create(
            messages=messages_to_send,
            model="llama-3.3-70b-versatile",
            temperature=1.2,
            max_tokens=32000,
            top_p=1,
            stop=None,
            stream=False,
        )
        response = chat_completion.choices[0].message.content.strip()
        log_message(f"[LLM] Groq response: {response}")
        if not response:
            raise ValueError("Empty response from Groq LLM.")
        log_message(f"[LLM] Received response from Groq for client {client_id}.")
        conversation_history[client_id].append({"role": "user", "content": transcript})
        conversation_history[client_id].append({"role": "assistant", "content": response})
        if len(conversation_history[client_id]) > MAX_HISTORY_MESSAGES:
            conversation_history[client_id] = conversation_history[client_id][-MAX_HISTORY_MESSAGES:]
        return response
    except Exception as e:
        log_message(f"[Error] Groq API call failed: {e}")
        return "Uh oh! My circuits are fuzzy. What did you say?"

def transcribe_audio_chunk(audio_data_np, sample_rate, client_id):
    global groq_client, client_tokens, pending_images
    if groq_client is None:
        return
    client_info = client_tokens.get(client_id)
    if not client_info:
        log_message(f"[Error] Could not find token/role info for client {client_id}. Cannot process audio.")
        return
    client_role = client_info.get("role", 1)
    log_message(f"[Transcription] Processing {len(audio_data_np) / sample_rate:.2f} seconds of audio for client {client_id} (Role: {client_role})...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    input_audio_filename = os.path.join(AUDIO_CHUNKS_DIR, f"chunk_{client_id}_{timestamp}.wav")
    start_time = time.time()
    try:
        if audio_data_np.dtype == np.float32:
            audio_data_float32 = audio_data_np
        else:
            audio_data_float32 = audio_data_np.astype(np.float32) / np.iinfo(DTYPE).max
        os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)
        sf.write(input_audio_filename, audio_data_float32, sample_rate)
        log_message(f"[+] Saved audio chunk to: {input_audio_filename}")
        with open(input_audio_filename, 'rb') as audio_file:
            audio_content = audio_file.read()
        language_code = client_tokens[client_id].get("language_code", "en")
        log_message(f"[Transcription] Using language code: {language_code}")
        files = {
            'file': ('audio.wav', audio_content, 'audio/wav'),
            'model': (None, 'whisper-large-v3'),
            'language': (None, language_code)
        }
        headers = {
            'Authorization': f'Bearer {os.environ.get("GROQ_API_KEY")}'
        }
        response = requests.post(
            'https://api.groq.com/openai/v1/audio/transcriptions',
            headers=headers,
            files=files
        )
        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")
        result = response.json()
        transcription_text = result.get('text', '').strip()
        end_transcription_time = time.time()
        transcription_time = end_transcription_time - start_time
        log_message("-" * 20)
        log_message(f"Client {client_id} said ({os.path.basename(input_audio_filename)}): {transcription_text}")
        log_message("-" * 20)

        tts_filename = f"response_{client_id}_{timestamp}.mp3"
        toy_response = ""
        
        with pending_images_lock:
            if client_id in pending_images and transcription_text:
                # Image analysis with transcription
                image_path = pending_images[client_id]
                log_message(f"[Image Analysis] Found stored image for client {client_id}: {image_path}")
                start_analysis_time = time.time()
                toy_response = analyze_image_with_llama(image_path, client_id, transcription_text)
                end_analysis_time = time.time()
                analysis_time = end_analysis_time - start_analysis_time
                log_message(f"[Image Analysis] Analysis completed in {analysis_time:.2f} seconds")

                # Generate TTS for analysis result
                start_tts_time = time.time()
                generate_tts_audio2(toy_response, tts_filename, client_id)
                end_tts_time = time.time()
                tts_time = end_tts_time - start_tts_time
                log_message(f"[TTS] TTS generated in {tts_time:.2f} seconds")

                # Send analysis notification
                notify_image_status(client_id, image_path, "imageanalyzed", toy_response)
                
                # Clean up stored image
                try:
                    os.remove(image_path)
                    log_message(f"[Image Cleanup] Deleted stored image: {image_path}")
                    del pending_images[client_id]
                except OSError as e:
                    log_message(f"[Image Cleanup Error] Failed to delete image {image_path}: {e}")
            elif transcription_text:
                # Normal conversation without image
                start_llm_time = time.time()
                toy_response = get_toy_response(transcription_text, client_id, client_role)
                end_llm_time = time.time()
                llm_time = end_llm_time - start_llm_time
                start_tts_time = time.time()
                generate_tts_audio2(toy_response, tts_filename, client_id)
                end_tts_time = time.time()
                tts_time = end_tts_time - start_tts_time
            else:
                log_message(f"[Transcription] No speech detected in the chunk from client {client_id}.")

        def delayed_cleanup(file_path, delay=10):
            time.sleep(delay)
            try:
                os.remove(file_path)
                log_message(f"[Delayed Cleanup] Deleted input audio file: {file_path}")
            except OSError as e:
                log_message(f"[Delayed Cleanup Error] Failed to delete input audio file {file_path}: {e}")
        cleanup_thread = threading.Thread(target=delayed_cleanup, args=(input_audio_filename,))
        cleanup_thread.start()
    except Exception as e:
        log_message(f"[Error] Main processing loop failed for {input_audio_filename}: {e}")

def process_audio():
    global last_transcription_time, audio_buffer
    while running:
        time.sleep(1)

def cleanup_audio_files(client_id):
    try:
        client_files = []
        for filename in os.listdir(OUTPUT_AUDIO_DIR):
            if filename.startswith(f"response_{client_id}_") and os.path.isfile(os.path.join(OUTPUT_AUDIO_DIR, filename)):
                client_files.append(filename)
        deleted_count = 0
        for filename in client_files:
            file_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
            try:
                os.remove(file_path)
                log_message(f"[Cleanup] Deleted audio file after client confirmation: {file_path}")
                deleted_count += 1
            except OSError as e:
                log_message(f"[Cleanup Error] Failed to delete {file_path}: {e}")
        if deleted_count > 0:
            log_message(f"[Cleanup] Removed {deleted_count} audio files for client {client_id} after playback confirmation")
    except Exception as e:
        log_message(f"[Cleanup Error] Error during audio cleanup for client {client_id}: {e}")

def cleanup_pending_images():
    while running:
        current_time = time.time()
        with pending_images_lock:
            for client_id in list(pending_images.keys()):
                image_path = pending_images[client_id]
                if os.path.exists(image_path):
                    file_time = os.path.getmtime(image_path)
                    if current_time - file_time > 300:  # 5 minutes
                        try:
                            os.remove(image_path)
                            del pending_images[client_id]
                            log_message(f"[Image Cleanup] Removed stale image for client {client_id}: {image_path}")
                        except OSError as e:
                            log_message(f"[Image Cleanup Error] Failed to remove stale image {image_path}: {e}")
        time.sleep(60)  # Check every minute
def cleanup_supabase_audio():
    supabase_url = "https://ozwvxxubkseztcakoobv.supabase.co"
    supabase_service_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96d3Z4eHVia3NlenRjYWtvb2J2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODcxNzc1OCwiZXhwIjoyMDU0MjkzNzU4fQ.7mdrn66Ma8dQiUdsTv8F5l4uEjqpd8gLi51hSevrEs8"
    if not supabase_service_key:
        log_message("[Supabase Cleanup Error] SUPABASE_SERVICE_KEY not found in environment variables")
        return
    supabase: Client = create_client(supabase_url, supabase_service_key)
    bucket_name = "toyresponse"
    max_age_seconds = 3600  # 1 hour
    global running

    while running:
        try:
            # List all files in the 'audio' directory recursively
            files = []
            # Fetch files for each client subdirectory
            client_dirs = supabase.storage.from_(bucket_name).list(path="audio")
            for client_dir in client_dirs:
                if client_dir.get("name"):  # Ensure it's a directory
                    client_files = supabase.storage.from_(bucket_name).list(path=f"audio/{client_dir['name']}")
                    for file in client_files:
                        file["full_path"] = f"audio/{client_dir['name']}/{file['name']}"
                        files.append(file)

            current_time = time.time()
            deleted_files = 0
            for file in files:
                if "metadata" in file and "lastModified" in file["metadata"]:
                    try:
                        last_modified = datetime.datetime.fromisoformat(
                            file["metadata"]["lastModified"].replace("Z", "+00:00")
                        ).timestamp()
                        if current_time - last_modified > max_age_seconds:
                            supabase.storage.from_(bucket_name).remove(file["full_path"])
                            log_message(f"[Supabase Cleanup] Deleted old audio file: {file['full_path']}")
                            deleted_files += 1
                    except ValueError as ve:
                        log_message(f"[Supabase Cleanup Error] Invalid lastModified format for {file['full_path']}: {ve}")
                else:
                    log_message(f"[Supabase Cleanup Warning] No metadata or lastModified for {file['full_path']}")
            log_message(f"[Supabase Cleanup] Processed {len(files)} files, deleted {deleted_files} old files")
        except Exception as e:
            log_message(f"[Supabase Cleanup Error] Failed to clean up audio files: {e}")
            if hasattr(e, 'response') and e.response:
                log_message(f"[Supabase Cleanup Error Details] Response: {e.response.text}")
        time.sleep(3600)  # Run every 1 hour, that also old i hour file

def start_udp_server():
    global running, audio_buffer
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((UDP_HOST, UDP_PORT))
            log_message(f"[*] UDP server listening on {UDP_HOST}:{UDP_PORT}")
            while running:
                s.settimeout(1.0)
                try:
                    data, addr = s.recvfrom(BUFFER_SIZE)
                    if data:
                        token_length = int.from_bytes(data[:4], 'big')
                        token = data[4:4+token_length].decode('utf-8')
                        log_message(f"[UDP] Received token: {token[:8]}...")
                        client_id = None
                        for cid, token_data in client_tokens.items():
                            if token_data["token"] == token:
                                client_id = cid
                                log_message(f"[UDP] Valid token received from client {client_id}")
                                break
                        if client_id is None:
                            log_message(f"[UDP] Invalid token received, rejecting packet from {addr}")
                            continue
                        if client_id not in audio_buffer:
                            audio_buffer[client_id] = queue.Queue()
                            log_message(f"[UDP] New client connected: {client_id} from {addr}")
                        payload = data[4+token_length:]
                        if payload.startswith(b"STOP"):
                            log_message(f"[UDP] Received STOP signal from client {client_id}")
                            if client_id in audio_buffer:
                                accumulated_audio = []
                                client_buffer = audio_buffer[client_id]
                                while not client_buffer.empty():
                                    accumulated_audio.append(client_buffer.get_nowait())
                                if accumulated_audio:
                                    full_audio_bytes = b''.join(accumulated_audio)
                                    try:
                                        audio_np = np.frombuffer(full_audio_bytes, dtype=DTYPE)
                                        if audio_np.size > 0:
                                            log_message(f"[UDP] Audio data converted to numpy array, transcribing...")
                                            transcribe_audio_chunk(audio_np, SAMPLE_RATE, client_id)
                                        else:
                                            log_message(f"[Processor] Empty audio chunk for client {client_id}")
                                    except Exception as e:
                                        log_message(f"[Processor Error] Processing failed for client {client_id}: {e}")
                                del audio_buffer[client_id]
                                log_message(f"[UDP] Removed client {client_id} from audio_buffer after STOP processing.")
                            continue
                        log_message(f"[UDP] Received audio chunk from {client_id} ({len(payload)} bytes)")
                        audio_buffer[client_id].put(payload)
                except socket.timeout:
                    continue
    except Exception as e:
        log_message(f"[UDP Server Error] An error occurred: {e}")
        running = False

def start_image_udp_server():
    global running, image_buffer
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((UDP_HOST, IMAGE_UDP_PORT))
            log_message(f"[*] Image UDP server listening on {UDP_HOST}:{IMAGE_UDP_PORT}")
            while running:
                s.settimeout(1.0)
                try:
                    data, addr = s.recvfrom(BUFFER_SIZE)
                    if data:
                        # Parse token
                        token_length = int.from_bytes(data[:4], 'big')
                        token = data[4:4+token_length].decode('utf-8')
                        log_message(f"[Image UDP] Received token: {token[:8]}... from {addr}")

                        # Validate token
                        client_id = None
                        for cid, token_data in client_tokens.items():
                            if token_data["token"] == token:
                                client_id = cid
                                log_message(f"[Image UDP] Valid token received from client {client_id}")
                                break
                        if client_id is None:
                            log_message(f"[Image UDP] Invalid token received, rejecting packet from {addr}")
                            continue

                        # Initialize client buffer
                        if client_id not in image_buffer:
                            image_buffer[client_id] = {}
                            log_message(f"[Image UDP] New client connected for images: {client_id} from {addr}")

                        # Parse payload
                        payload = data[4+token_length:]
                        if payload.startswith(b"IMAGE"):
                            image_id = payload[5:41].decode('utf-8')  # 36-byte UUID
                            chunk_idx = int.from_bytes(payload[41:45], 'big')
                            total_chunks = int.from_bytes(payload[45:49], 'big')
                            chunk_data = payload[49:]
                            
                            # Initialize image buffer for this image_id
                            if image_id not in image_buffer[client_id]:
                                image_buffer[client_id][image_id] = [None] * total_chunks
                            # Store chunk
                            image_buffer[client_id][image_id][chunk_idx] = chunk_data
                            log_message(f"[Image UDP] Received IMAGE chunk {chunk_idx + 1}/{total_chunks} for image ID {image_id} from client {client_id}")
                            continue

                        elif payload.startswith(b"IMEND"):
                            image_id = payload[5:41].decode('utf-8')  # 36-byte UUID
                            total_chunks = int.from_bytes(payload[45:49], 'big')
                            if client_id in image_buffer and image_id in image_buffer[client_id]:
                                chunks = image_buffer[client_id][image_id]
                                if all(chunk is not None for chunk in chunks):
                                    # Concatenate chunks (base64-encoded)
                                    full_image_data = b''.join(chunks)
                                    try:
                                        # Save the image (handles base64 decoding and MQTT notification)
                                        save_received_image(full_image_data, client_id, image_id)
                                        log_message(f"[Image UDP] Successfully reassembled and saved image ID {image_id} for client {client_id}")
                                    except Exception as e:
                                        log_message(f"[Image UDP] Failed to save image ID {image_id} for client {client_id}: {e}")
                                        notify_image_status(client_id, f"failed_image_{image_id}", "imagereceived", f"Error: {str(e)}")
                                    # Clean up buffer
                                    del image_buffer[client_id][image_id]
                                    if not image_buffer[client_id]:
                                        del image_buffer[client_id]
                                else:
                                    log_message(f"[Image UDP] Missing chunks for image ID {image_id} from client {client_id}")
                                    notify_image_status(client_id, f"failed_image_{image_id}", "imagereceived", "Missing chunks")
                            else:
                                log_message(f"[Image UDP] No buffer found for image ID {image_id} from client {client_id}")
                            continue

                        log_message(f"[Image UDP] Received unknown packet type from {client_id} ({len(payload)} bytes)")
                except socket.timeout:
                    continue
                except Exception as e:
                    log_message(f"[Image UDP] Error processing packet from {addr}: {e}")
    except Exception as e:
        log_message(f"[Image UDP Server Error] An error occurred: {e}")
        running = False

def start_server():
    global running, groq_client, http_server_thread
    load_dotenv()
    log_message("[*] Loaded environment variables from .env file.")
    # Start Supabase cleanup thread
    cleanup_supabase_thread = threading.Thread(target=cleanup_supabase_audio, daemon=True)
    cleanup_supabase_thread.start()
    log_message("[Server] Supabase cleanup thread started")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log_message("[Error] GROQ_API_KEY not found.")
        sys.exit(1)
    try:
        groq_client = Groq(api_key=api_key)
        log_message("[*] Groq client initialized.")
    except Exception as e:
        log_message(f"[Error] Failed to initialize Groq client: {e}")
        sys.exit(1)
    class ServerThread(threading.Thread):
        def __init__(self, app, host, port):
            threading.Thread.__init__(self)
            self.server = make_server(host, port, app)
            self.ctx = app.app_context()
            self.ctx.push()
        def run(self):
            log_message(f"[*] HTTP server started on {self.server.host}:{self.server.port}")
            self.server.serve_forever()
        def shutdown(self):
            self.server.shutdown()
            log_message("[*] HTTP server stopped.")
    # Add these helper functions
    def get_file_time(filename):
        try:
            filepath = os.path.join(RECEIVED_IMAGES_DIR, filename)
            timestamp = os.path.getmtime(filepath)
            return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            log_message(f"[Error] Failed to get file time for {filename}: {e}")
            return "Unknown"

    def get_user_from_filename(filename):
            # Extract client ID from filename (format: image_{client_id}_{image_id}_{timestamp}.jpeg)
            parts = filename.split('_')
            if len(parts) > 2:
                return parts[1]  # client_id is the second part
            return "Unknown"

    flask_app = Flask(__name__)
    
    @flask_app.route('/audio/<path:filename>')
    def serve_audio(filename):
        return send_from_directory(OUTPUT_AUDIO_DIR, filename)
    
    @flask_app.route('/')
    def show_images():
        try:
            # List all files in RECEIVED_IMAGES_DIR
            image_files = [f for f in os.listdir(RECEIVED_IMAGES_DIR) 
                        if f.lower().endswith(('.jpeg', '.jpg', '.png', '.gif'))]
            # Sort by modification time (newest first)
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(RECEIVED_IMAGES_DIR, x)), reverse=True)
            
            # Add the helper functions to the template context
            return render_template('images.html', 
                                images=image_files,
                                get_file_time=get_file_time,
                                get_user_from_filename=get_user_from_filename)
        except Exception as e:
            log_message(f"[Error] Failed to list images: {e}")
            return "Error loading images", 500

    @flask_app.route('/images/<path:filename>')
    def serve_image(filename):
        return send_from_directory(RECEIVED_IMAGES_DIR, filename)

    @flask_app.route('/default_audio/<path:filename>')
    def serve_default_audio(filename):
        return send_from_directory('default_audio', filename)
    # New endpoint for toy updates
    @flask_app.route('/update_toy', methods=['POST', 'OPTIONS'])
    def update_toy():
        # Handle OPTIONS request
        if request.method == 'OPTIONS':
            return {}, 200
            
        try:
            data = request.get_json()
            if not data:
                log_message("[Update Toy] No JSON data provided in request")
                return jsonify({"error": "No JSON data provided"}), 400

            serial_number = data.get('serialNumber')
            role_type = data.get('roleType')
            language = data.get('language')
            voice = data.get('voice')

            if not all([serial_number, role_type, language, voice]):
                log_message("[Update Toy] Missing required fields in request")
                return jsonify({"error": "Missing required fields"}), 400

            # Optional: Validate token or add authentication
            token = data.get('token')
            if token and not is_valid_token(token, serial_number):
                log_message(f"[Update Toy] Invalid token for client {serial_number}")
                return jsonify({"error": "Invalid or expired token"}), 401

            # Validate toy exists
            if not is_valid_toy(serial_number):
                log_message(f"[Update Toy] Toy with serial_number {serial_number} not found")
                return jsonify({"error": "Toy not found"}), 404

            # Publish MQTT message
            msg_id = str(uuid.uuid4())
            message = {
                "msgId": msg_id,
                "identifier": "updaterole",
                "outParams": {
                    "roleType": role_type,
                    "language": language,
                    "voice": voice
                }
            }
            topic = f"user/cheekotoy/{serial_number}/thing/data/post"
            if mqtt_client_server and mqtt_client_server.is_connected():
                payload_str = json.dumps(message)
                result = mqtt_client_server.publish(topic, payload_str, qos=1)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    log_message(f"[Update Toy] Successfully published update for {serial_number} to {topic}")
                    return jsonify({"success": True, "msgId": msg_id}), 200
                else:
                    log_message(f"[Update Toy] Failed to publish update for {serial_number}. RC: {result.rc}")
                    return jsonify({"error": "Failed to publish MQTT message"}), 500
            else:
                log_message("[Update Toy] MQTT client not connected")
                return jsonify({"error": "MQTT server not connected"}), 500

        except Exception as e:
            log_message(f"[Update Toy Error] Failed to process update request: {e}")
            return jsonify({"error": f"Server error: {str(e)}"}), 500
        
    log_message("[Debug] Attempting to define /dashboard route...")
    @flask_app.route('/dashboard')
    def dashboard():
        """Displays connected devices and recent logs."""
        connected_devices = list(client_tokens.keys()) # Get list of client IDs
        # Get the last 10 log messages (or fewer if less than 10 exist)
        logs_to_display = list(log_messages)[-20:]
        logs_to_display.reverse() # Show newest first
        return render_template('dashboard.html',
                               devices=connected_devices,
                               logs=logs_to_display)

    @flask_app.route('/audio_dashboard')
    def audio_dashboard():
        """Displays a list of generated audio files."""
        audio_files_data = []
        if not os.path.exists(OUTPUT_AUDIO_DIR):
            log_message(f"[Audio Dashboard] Directory not found: {OUTPUT_AUDIO_DIR}")
            return render_template('audio_dashboard.html', audio_files=audio_files_data)

        try:
            for filename in os.listdir(OUTPUT_AUDIO_DIR):
                if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.aac')): # Add more extensions if needed
                    filepath = os.path.join(OUTPUT_AUDIO_DIR, filename)
                    try:
                        stat_info = os.stat(filepath)
                        size_kb = round(stat_info.st_size / 1024, 2)
                        modified_time = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        audio_files_data.append({
                            "name": filename,
                            "size_kb": size_kb,
                            "modified_time": modified_time
                        })
                    except Exception as e:
                        log_message(f"[Audio Dashboard] Error getting info for file {filename}: {e}")
            # Sort by modification time (newest first)
            audio_files_data.sort(key=lambda x: x['modified_time'], reverse=True)
        except Exception as e:
            log_message(f"[Audio Dashboard] Error listing audio files: {e}")

        return render_template('audio_dashboard.html', audio_files=audio_files_data)

    @flask_app.route('/delete_all_images', methods=['POST'])
    def delete_all_images():
        """Deletes all image files from the RECEIVED_IMAGES_DIR."""
        if not os.path.exists(RECEIVED_IMAGES_DIR):
            log_message(f"[Delete All Images] Directory not found: {RECEIVED_IMAGES_DIR}")
            return redirect(url_for('show_images'))

        deleted_count = 0
        error_count = 0
        try:
            for filename in os.listdir(RECEIVED_IMAGES_DIR):
                filepath = os.path.join(RECEIVED_IMAGES_DIR, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.unlink(filepath)
                        log_message(f"[Delete All Images] Deleted file: {filepath}")
                        deleted_count += 1
                except Exception as e:
                    log_message(f"[Delete All Images] Error deleting file {filepath}: {e}")
                    error_count += 1
            
            log_message(f"[Delete All Images] Deleted {deleted_count} files. Encountered {error_count} errors.")
        except Exception as e:
            log_message(f"[Delete All Images] Error accessing directory {RECEIVED_IMAGES_DIR}: {e}")

        return redirect(url_for('show_images'))

    @flask_app.route('/delete_all_audio', methods=['POST'])
    def delete_all_audio():
        """Deletes all audio files from the OUTPUT_AUDIO_DIR."""
        if not os.path.exists(OUTPUT_AUDIO_DIR):
            log_message(f"[Delete All Audio] Directory not found: {OUTPUT_AUDIO_DIR}")
            # Optionally, add a flash message here for the user
            return redirect(url_for('audio_dashboard'))

        deleted_count = 0
        error_count = 0
        try:
            for filename in os.listdir(OUTPUT_AUDIO_DIR):
                filepath = os.path.join(OUTPUT_AUDIO_DIR, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath): # Check if it's a file or a symlink
                        os.unlink(filepath) # Use unlink to remove files and symlinks
                        log_message(f"[Delete All Audio] Deleted file: {filepath}")
                        deleted_count += 1
                    # Optionally, handle subdirectories if they are not expected or should also be deleted
                except Exception as e:
                    log_message(f"[Delete All Audio] Error deleting file {filepath}: {e}")
                    error_count += 1
            
            log_message(f"[Delete All Audio] Deleted {deleted_count} files. Encountered {error_count} errors.")
            # Optionally, add a flash message here for the user about success/failure counts
        except Exception as e:
            log_message(f"[Delete All Audio] Error accessing directory {OUTPUT_AUDIO_DIR}: {e}")
            # Optionally, add a flash message here for the user

        return redirect(url_for('audio_dashboard'))

    def setup_mqtt_server():
        global mqtt_client_server
        try:
            # Add connection retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    mqtt_client_server = mqtt.Client(
                        mqtt.CallbackAPIVersion.VERSION2,
                        client_id=MQTT_CLIENT_ID_SERVER,
                        protocol=mqtt.MQTTv5
                    )
                    mqtt_client_server.on_connect = on_connect_server
                    mqtt_client_server.on_message = on_message_server
                    
                    # Add connection logging
                    log_message(f"[MQTT] Attempting to connect to broker at {MQTT_BROKER}:{MQTT_PORT} (attempt {attempt + 1}/{max_retries})")
                    mqtt_client_server.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                    mqtt_client_server.loop_start()
                    log_message("[MQTT] Successfully connected to broker")
                    return True
                    
                except Exception as e:
                    log_message(f"[MQTT] Connection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
                    raise
                    
        except Exception as e:
            log_message(f"[MQTT Server] Error connecting to MQTT broker: {e}")
            mqtt_client_server = None
            return False
    setup_mqtt_server()
    if not mqtt_client_server:
        log_message("[Error] MQTT client setup failed. Exiting.")
        sys.exit(1)
    http_server_thread = ServerThread(flask_app, HTTP_HOST, HTTP_PORT)
    http_server_thread.daemon = True
    http_server_thread.start()
    processor_thread = threading.Thread(target=process_audio, daemon=True)
    processor_thread.start()
    udp_thread = threading.Thread(target=start_udp_server, daemon=True)
    udp_thread.start()
    image_udp_thread = threading.Thread(target=start_image_udp_server, daemon=True)
    image_udp_thread.start()
    cleanup_thread = threading.Thread(target=cleanup_pending_images, daemon=True)
    cleanup_thread.start()
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        log_message("\n[*] Server shutting down initiated by user...")
        running = False
    log_message("[*] Stopping MQTT loop...")
    if mqtt_client_server:
        mqtt_client_server.loop_stop()
        mqtt_client_server.loop_stop()
        mqtt_client_server.disconnect()
    log_message("[*] Waiting for threads to finish...")
    if http_server_thread:
        http_server_thread.shutdown()
    if udp_thread.is_alive():
        udp_thread.join(timeout=2)
    if image_udp_thread.is_alive():
        image_udp_thread.join(timeout=2)
    if processor_thread.is_alive():
        processor_thread.join(timeout=2)
    if cleanup_thread.is_alive():
        cleanup_thread.join(timeout=2)
    if http_server_thread and http_server_thread.is_alive():
        http_server_thread.join(timeout=2)
    log_message("[*] Server stopped.")

def is_valid_toy(client_id):
    SUPABASE_URL = "https://popppjirsdedxhetcphs.supabase.co"
    SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBvcHBwamlyc2RlZHhoZXRjcGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM3NjMxMDAsImV4cCI6MjA1OTMzOTEwMH0.Ihv60cbfUSeDN5dPDsOZRz4y79ek3D5YZZgKwBsMkSc"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{SUPABASE_URL}/rest/v1/toys?serial_number=eq.{client_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200 and resp.json():
        return True
    return False

def get_toy_role_type(client_id):
    SUPABASE_URL = "https://popppjirsdedxhetcphs.supabase.co"
    SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBvcHBwamlyc2RlZHhoZXRjcGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM3NjMxMDAsImV4cCI6MjA1OTMzOTEwMH0.Ihv60cbfUSeDN5dPDsOZRz4y79ek3D5YZZgKwBsMkSc"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/toys?serial_number=eq.{client_id}&select=role_type"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            log_message(f"[Supabase Error] Failed to fetch role_type: {resp.status_code} - {resp.text}")
            return None
        data = resp.json()
        if not data:
            log_message(f"[Supabase] No toy found with serial_number: {client_id}")
            return None
        role_type = data[0].get("role_type")
        log_message(f"[Supabase] Retrieved role_type '{role_type}' for client {client_id}")
        return role_type
    except Exception as e:
        log_message(f"[Supabase Error] Exception while fetching role_type: {e}")
        return None

def get_toy_voice(client_id):
    SUPABASE_URL = "https://popppjirsdedxhetcphs.supabase.co"
    SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBvcHBwamlyc2RlZHhoZXRjcGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM3NjMxMDAsImV4cCI6MjA1OTMzOTEwMH0.Ihv60cbfUSeDN5dPDsOZRz4y79ek3D5YZZgKwBsMkSc"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/toys?serial_number=eq.{client_id}&select=voice"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            log_message(f"[Supabase Error] Failed to fetch voice: {resp.status_code} - {resp.text}")
            return None
        data = resp.json()
        if not data:
            log_message(f"[Supabase] No toy found with serial_number: {client_id}")
            return None
        voice = data[0].get("voice")
        log_message(f"[Supabase] Retrieved voice '{voice}' for client {client_id}")
        return voice
    except Exception as e:
        log_message(f"[Supabase Error] Exception while fetching voice: {e}")
        return None

def get_toy_language(client_id):
    SUPABASE_URL = "https://popppjirsdedxhetcphs.supabase.co"
    SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBvcHBwamlyc2RlZHhoZXRjcGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM3NjMxMDAsImV4cCI6MjA1OTMzOTEwMH0.Ihv60cbfUSeDN5dPDsOZRz4y79ek3D5YZZgKwBsMkSc"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/toys?serial_number=eq.{client_id}&select=language"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            log_message(f"[Supabase Error] Failed to fetch language: {resp.status_code} - {resp.text}")
            return None
        data = resp.json()
        if not data:
            log_message(f"[Supabase] No toy found with serial_number: {client_id}")
            return None
        language = data[0].get("language")
        log_message(f"[Supabase] Retrieved language '{language}' for client {client_id}")
        return language
    except Exception as e:
        log_message(f"[Supabase Error] Exception while fetching language: {e}")
        return None

if __name__ == "__main__":
    start_server()