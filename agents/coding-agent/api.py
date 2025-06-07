import os
from typing import Optional, Union
from together import Together
from anthropic import Anthropic 

def api(
        user_prompt: str, 
        client: Optional[Union[Together, Anthropic]] = None,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", 
        system_prompt: str = "You are helpful language model assistant, designed to produce factual and correct responses to queries.", 
    ):
    try:
        # fallback to together if no client provided
        if client is None:
            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            client = Together(api_key=api_key)
        
        # anthropic has different api structure than together
        if isinstance(client, Anthropic):
            response = client.messages.create(
                model="claude-4-sonnet-20250514",  # hardcoded to latest claude model
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt, 
                    }
                ]
            )
            return response.content[0].text
        
        # together uses openai-compatible chat completions format
        response = client.chat.completions.create(
            model=model_name,  # model_name ignored for anthropic, used for together
            max_tokens=4000,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt, 
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        return None  # graceful degradation instead of crashing
