from openai import OpenAI
import os

_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")

CLIENT = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def dummy_llm() -> str:
    """Dummy LLM call which returns a standard string. Used as fallback for exceptions.

    Args:
        prompt: User prompt.
        lore: Fantasy lore snippet from RAG.

    Returns:
        A fantasy-styled answer string.
    """
    return (
        "Alas, in this mystical prototype the ancient spirits have yet to awaken "
        "their full wisdom. Return soon, and deeper secrets shall be revealed!"
    )

def deepseek_chat(base_prompt: str, user_prompt: str) -> str:
    """Calls the DeepSeek API to generate a response to the user's query.

    Args:
        base_prompt: The base prompt to use for the query.
        user_prompt: The user's query.

    Returns:
        The response from the DeepSeek API.
    """

    response = CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "assistant", "content": base_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content
