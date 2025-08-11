from openai import OpenAI
from core.RAG import RAG as RAG
from core.ImageGenerator import Generator

with open("API.txt", "r") as f:
    api_key = f.read().strip()

CLIENT = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_LLM(prompt, query):
    response = CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": query},
        ],
        stream=False
    )

    return response.choices[0].message.content

def translate_query(query):

    with open("translation_prompt.txt", "r") as f:
        translation_prompt = f.read().strip()
    translated = call_LLM(query=query, prompt=translation_prompt)

    if translated.strip().upper().startswith("OUT-OF-SCOPE:"):
        # Out of scope detected, use 422 Unprocessable Entity
        return translated, 422
    else:
        # Valid translation, 200 OK
        return translated, 200

def call_LLM_output(context):

    with open("output_prompt.txt", "r", encoding='utf-8') as f:
        output_prompt = f.read().strip()

    return call_LLM(query=context, prompt=output_prompt)

def generate_outputs(context, max_retries=3):
    with open("output_prompt.txt", "r", encoding='utf-8') as f:
        output_prompt = f.read().strip()


    for attempt in range(max_retries):

        LLM_output = call_LLM(query=context, prompt=output_prompt)
        parts = LLM_output.split("===STABLE DIFFUSION PROMPT===")

        if len(parts) == 2:
            description = parts[0].strip()
            sd_prompt = parts[1].strip()
            return description, sd_prompt
        else:
            print("warning: malformed output, retrying")
            pass

    # After retries exhausted, fallback
    fallback_description = "Description unavailable due to malformed LLM output."
    return fallback_description, None

def generate_image(generator, sd_prompt):
    output_file = "output.png"
    lora_file = "FantasyClassics.safetensors"

    with open("SD_keywords.txt", "r") as f:
        keywords = f.read().strip()

    full_prompt = keywords + sd_prompt

    #generator.load_lora(lora_file)

    image = generator.generate(full_prompt)

    image.save(output_file)

    return image


def main():

    retriever = RAG()
    generator = Generator()

    query = "show me a person in the tavern drinking something"

    # preprocess query
    translated, code = translate_query(query)

    if code == 422:
        print("Query out of scope, skipping retrieval and image generation.")
        print("Assistant response:", translated)

        # Optionally, generate a fallback image or use a default description/prompt
        fallback_description = translated
        fallback_image = ...
        return

    # retrieve context
    full_context = retriever.answer_question(translated)

    output_description, SD_prompt = generate_outputs(full_context)

    if SD_prompt is not None:
        generate_image(generator, SD_prompt)

    # display description + image

    print("\n#### Full Context Sent to LLM ####")
    print(full_context)

    print("\n#### Question ####")
    print(query)

    print("\n#### Description ####")
    print(output_description)

    print("\n#### prompt ####")
    print(SD_prompt)


if __name__ == "__main__":
    main()