from openai import OpenAI
import os

_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

CLIENT = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


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
'''
def deepseek_chat(prompt: str, lore: str, api_key: str | None = None, **kwargs) -> str:
    """Call DeepSeek Cloud chat completion API.

    Not wired yet; kept for future expansion.
    """
    import os, httpx, json

    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",  # adjust if using a different model name
        "messages": [
            {"role": "system", "content": f"You are an oracle in a fantasy realm. Use the following lore to answer: {lore}"},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = httpx.post(_API_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # DeepSeek schema similar to OpenAI: choices[0].message.content
        return data["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        # Bubble up an error so caller can fallback
        raise RuntimeError(f"DeepSeek API error: {exc}")

'''
# TODO handle key not set
def deepseek_chat(base_prompt, user_prompt):
    response = CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "assistant", "content": base_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content
