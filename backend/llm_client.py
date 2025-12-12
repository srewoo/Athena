"""
Simple LLM client wrapper
"""
import time
from typing import Optional


class LLMClient:
    """Unified LLM client"""
    
    def __init__(self):
        pass
    
    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> dict:
        """Execute chat completion"""
        start_time = time.time()
        
        try:
            if provider == "openai":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=api_key)
                model = model_name or "gpt-4o"
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "output": response.choices[0].message.content,
                    "error": None,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
                
            elif provider == "claude":
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=api_key)
                model = model_name or "claude-3-5-sonnet-20241022"
                
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=temperature if temperature > 0 else 0.01
                )
                
                return {
                    "output": response.content[0].text,
                    "error": None,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
                }
                
            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = model_name or "gemini-2.0-flash-exp"
                
                genai_model = genai.GenerativeModel(model)
                full_prompt = f"{system_prompt}\n\n{user_message}"
                
                response = await genai_model.generate_content_async(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                return {
                    "output": response.text,
                    "error": None,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "tokens_used": 0
                }
                
            else:
                return {
                    "output": "",
                    "error": f"Unsupported provider: {provider}",
                    "latency_ms": 0,
                    "tokens_used": 0
                }
                
        except Exception as e:
            return {
                "output": "",
                "error": str(e),
                "latency_ms": int((time.time() - start_time) * 1000),
                "tokens_used": 0
            }


def get_llm_client():
    """Get LLM client instance"""
    return LLMClient()
