from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from uuid import uuid4
from pydantic import BaseModel
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from core.RAG import RAG as RAG
import logging
import os


load_dotenv()

from core.llm import dummy_llm, deepseek_chat
from core.RAG import dummy_rag
from core.ImageGenerator import Generator

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


app = FastAPI()

generator = Generator()
retriever = RAG()

# Allow frontend dev server to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# resources path
resources_dir = Path(__file__).parent / "resources"
resources_dir.mkdir(exist_ok=True)
app.mount("/resources", StaticFiles(directory=resources_dir), name="resources")

# output path
output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

placeholder_rel_path = "placeholder.png"

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    id: str
    answer: str
    image_url: str

class HistoryItem(BaseModel):
    id: str
    prompt: str
    answer: str
    image_url: str

history: dict[str, HistoryItem] = {}

def generate_image(generator: Generator, image_prompt: str, filename: str) -> str:
    ''' Generate an image by calling Stable Diffusion and save it locally.

    Args:
        generator: An instance of the diffuser generator class.
        image_prompt: The image prompt to generate.
        filename: The name to save the file to. Usually a UUID.

    Returns:
        The filename of the generated image. Used to retrieve the image later in frontend.
    '''
    output_file = filename+".png"
    output_path = f"./outputs/{output_file}"

    with open("SD_keywords.txt", "r") as f:
        keywords = f.read().strip()

    full_prompt = keywords + image_prompt

    negative_prompt = "EasyNegative, portrait, ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, bad quality, medium quality"

    #generator.load_lora(lora_file)
    logger.debug(f"saved image to {os.path.abspath(output_path)}")

    image = generator.generate(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            steps=55,
            cfg_scale=7,
        )

    image.save(output_path)

    return output_file

def translate_query(query: str) -> tuple[str, int]:
    """Calls an LLM to translate the raw user question. 
    The LLM will also validate the query by detecting if it is out of scope.

    Args:
        query: Raw user query.

    Returns:
        tuple of (translated_text, status_code) where code status is 200 if translation was successful, 422 if out of scope.
    """

    with open("translation_prompt.txt", "r") as f:
        translation_prompt = f.read().strip()
    translated = deepseek_chat(user_prompt=query, base_prompt=translation_prompt)

    if translated.strip().upper().startswith("OUT-OF-SCOPE:"):
        # Out of scope detected
        return translated, 422
    else:
        # Valid translation, 200 OK
        return translated, 200

def generate_outputs(rag_context: str, max_retries: int = 3) -> tuple[str, str | None]:
    """Iteratively calls the LLM to generate a description for the scene and an image prompt.
    
    Args:
        rag_context: The context retrieved from RAG.
        max_retries: Maximum number of retries.
    
    Returns:
        tuple of (description, image_prompt)
    """
    with open("output_prompt.txt", "r", encoding='utf-8') as f:
        image_generator_prompt = f.read().strip()


    for attempt in range(max_retries):

        LLM_output = deepseek_chat(user_prompt=rag_context, base_prompt=image_generator_prompt)
        parts = LLM_output.split("===STABLE DIFFUSION PROMPT===")

        if len(parts) == 2:
            description = parts[0].strip()
            sd_prompt = parts[1].strip()

            print("## Final outputs")
            print(description)
            print(sd_prompt)

            return description, sd_prompt
        else:
            print("warning: malformed output, retrying")
            pass

    # After retries exhausted, fallback
    fallback_description = "Description unavailable due to malformed LLM output."
    return fallback_description, None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Handle chat requests and provides a response to the frontend. Generates a response and an image based on the user's query.
    
    """

    description = ""
    image_url = ""

    item_id = str(uuid4())

    logger.debug(f"Invoked chat")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # preprocess query
    translated, code = translate_query(prompt)

    if code == 422:
        print("Query out of scope, skipping retrieval and image generation.")
        print("Assistant response:", translated)

        # TODO generate a better placeholder image
        description = translated
        image_url = f"/resources/{placeholder_rel_path}"

        return ChatResponse(id=item_id, answer=description, image_url=image_url)

    # retrieve context
    full_context = retriever.answer_question(query=translated, original_question=prompt)

    image_prompt = None

    try:
        description, image_prompt = generate_outputs(full_context)
    except RuntimeError as exc:
        logger.debug(f"DeepSeek unavailable, falling back to dummy: {exc}")
        description = dummy_llm()



    if image_prompt is not None:
        image_url = generate_image(generator=generator, image_prompt=image_prompt, filename=str(item_id))
        image_url = f"/outputs/{image_url}"
    else:
        image_url = f"/resources/{placeholder_rel_path}"

    history[item_id] = HistoryItem(
        id=item_id,
        prompt=prompt,
        answer=description,
        image_url=image_url,
    )

    return ChatResponse(id=item_id, answer=description, image_url=image_url)

@app.get("/history/{item_id}", response_model=HistoryItem)
async def get_history_item(item_id: str) -> HistoryItem:
    """Retrieve a specific chat history item by ID."""
    if item_id not in history:
        raise HTTPException(status_code=404, detail="Item not found")
    return history[item_id]