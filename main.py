#!/usr/bin/env python3

import os
import sys
import argparse
from dotenv import load_dotenv
import requests
import tiktoken
from typing import List, Dict, Optional

SYSTEM_PROMPT = """
"You are a helpful, friendly, and knowledgeable AI assistant. 
You provide clear, accurate, and concise responses while maintaining a conversational tone.
Keep in mind that user communicates with you through a command-line application.
"""

MAX_CONTEXT_TOKENS = 32000

class TokenCounter:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_text(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        return sum(self.count_text(msg["content"]) for msg in messages)

class DeepseekChat:
    def __init__(self, api_key: str, show_tokens: bool = False):
        self.api_key = api_key
        self.show_tokens = show_tokens
        self.token_counter = TokenCounter()
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def _make_api_request(self, temperature: float, max_tokens: int) -> str:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        current_tokens = self.token_counter.count_messages(self.messages)
        if current_tokens + max_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"Total tokens ({current_tokens} + {max_tokens}) would exceed "
                f"context window of {MAX_CONTEXT_TOKENS}"
            )
        
        data = {
            "model": "deepseek-chat",
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Deepseek API: {e}")
            sys.exit(1)
    
    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
    
    def get_response(self, temperature: float, max_tokens: int) -> Optional[str]:
        if self.show_tokens:
            current_tokens = self.token_counter.count_messages(self.messages)
            print(f"\nCurrent context size: {current_tokens} tokens")
        
        try:
            response = self._make_api_request(temperature, max_tokens)
            self.messages.append({"role": "assistant", "content": response})
            
            if self.show_tokens:
                new_tokens = self.token_counter.count_messages(self.messages)
                print(f"Response added {new_tokens - current_tokens} tokens")
                print(f"Total context size: {new_tokens} tokens")
            
            return response
        except ValueError as e:
            print(f"\nWarning: {e}")
            print("Consider starting a new conversation or summarizing the context.")
            return None

def load_api_key() -> str:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Load .env from the script's directory
    load_dotenv(os.path.join(script_dir, '.env'))
    api_key = os.getenv('SK_TOKEN')
    if not api_key:
        print("Error: SK_TOKEN not found in .env file")
        sys.exit(1)
    return api_key

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Chat with Deepseek LLM from command line')
    parser.add_argument('prompt', nargs='+', help='Initial prompt to start the conversation')
    parser.add_argument('-t', '--temperature', type=float, default=0.7,
                      help='Temperature for response generation (0.0-1.0, default: 0.7)')
    parser.add_argument('-m', '--max-tokens', type=int, default=2000,
                      help='Maximum tokens in response (default: 2000)')
    parser.add_argument('--show-tokens', action='store_true',
                      help='Show token usage information')
    return parser.parse_args()

def chat_loop(chat: DeepseekChat, args: argparse.Namespace) -> None:
    try:
        while True:
            response = chat.get_response(args.temperature, args.max_tokens)
            if response:
                print("\nDeepseek:", response)
            
            print("\nYou (press Ctrl+C to exit):", end=" ")
            user_input = input().strip()
            
            if user_input:
                chat.add_user_message(user_input)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

def main():
    args = parse_args()
    api_key = load_api_key()
    
    chat = DeepseekChat(api_key, args.show_tokens)
    chat.add_user_message(" ".join(args.prompt))
    
    chat_loop(chat, args)

if __name__ == "__main__":
    main() 