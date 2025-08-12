from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob
import chromadb
from chromadb.config import Settings
import uuid
from chromadb import PersistentClient
import re

LORE_LOCATION = "./scene"
DB_DIR = "./chroma_db"

# TODO incorperate this in class below and use as part of error/exception fallback
def dummy_rag(prompt: str) -> str:
    return (
        "In the age of ember-born dragons, the crystal city of Altheria floated "
        "above silver clouds, sustained by ancient runes and the songs of sky-magi."
    )

class RAG:

    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)

        chroma_client = PersistentClient(path=DB_DIR)

        self.collection = chroma_client.get_or_create_collection(name="my_collection")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading lore...")
        lore_chunks, metadata = self.load_markdown_files_new(LORE_LOCATION, exclude_subfolders=["./people", "./props"])
        embeddings = self.model.encode(lore_chunks).tolist()

        # Add to Chroma (skip if already populated)
        existing_count = self.collection.count()
        if existing_count == 0:
            print("Storing lore in ChromaDB...")
            ids = [str(uuid.uuid4()) for _ in range(len(lore_chunks))]
            self.collection.add(
                documents=lore_chunks,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
        else:
            print(f"ChromaDB already has {existing_count} documents.")

    # read .md files
    def load_markdown_files(self, folder_path):
        md_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
        chunks = []
        metadatas = []
        for filepath in md_files:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                for i, paragraph in enumerate(paragraphs):
                    chunks.append(paragraph)
                    metadatas.append({
                        "source": os.path.basename(filepath),
                        "chunk_index": i
                    })
        return chunks, metadatas


    def load_markdown_files_new(self, folder_path, exclude_subfolders=None):
        md_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
        chunks = []
        metadatas = []

        # Normalize excluded subfolder paths
        if exclude_subfolders:
            exclude_subfolders = [
                os.path.abspath(os.path.normpath(os.path.join(folder_path, ex)))
                for ex in exclude_subfolders
            ]

        #print(exclude_subfolders)

        for filepath in md_files:
            abs_filepath = os.path.abspath(filepath)

            # Check if file is in an excluded subfolder
            if exclude_subfolders:
                for ex_path in exclude_subfolders:
                    if os.path.commonpath([abs_filepath, ex_path]) == ex_path:
                        print(f"[Excluding] {abs_filepath} (in {ex_path})")
                        break
                else:
                    pass  # no exclusion matched, continue below
                    # continue below
            else:
                ex_path = None  # just for clarity

            # Skip if excluded
            if exclude_subfolders and any(os.path.commonpath([abs_filepath, ex]) == ex for ex in exclude_subfolders):
                continue

            rel_path = os.path.relpath(filepath, folder_path)
            folder = rel_path.split(os.sep)[0]  # top-level folder like "scene"

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

                # Paragraph-level chunks
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                for i, paragraph in enumerate(paragraphs):
                    matches = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", paragraph)
                    refs = ";".join(match[1] for match in matches) if matches else ""

                    chunks.append(paragraph)
                    metadatas.append({
                        "source": rel_path,
                        "chunk_index": i,
                        "folder": folder,
                        "type": "chunk",
                        "references": refs
                    })

        print(f"[Loaded] {len(chunks)} chunks from {len(md_files)} files (excluding subfolders)")
        return chunks, metadatas

    def get_chunks_by_filename(self, filename):
        return self.collection.get(where={"source": filename})

    def answer_question(self, query, original_question, top_k=3):
        question_embedding = self.model.encode([query]).tolist()
        # Get top-k results
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved_docs = results["documents"][0]
        retrieved_meta = results["metadatas"][0]
        distances = results["distances"][0]

        print(distances)

        linked_files = set()
        for meta in retrieved_meta:
            links = meta.get("references", "")
            linked_files.update(ref.strip() for ref in links.split(";") if ref.strip())

        linked_chunks = {}

        for fname in linked_files:
            full_path = os.path.join(LORE_LOCATION, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    linked_chunks[fname] = content
            except FileNotFoundError:
                print(f"[Missing File] Could not find referenced file: {full_path}")

        #TODO verbose logging
        print("\n--- RAG Prompt Breakdown ---")

        print("\n> Retrieved Lore Chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  Lore {i+1}: {doc}...")

        print("\n> Referenced Files Added:")
        for fname, content in linked_chunks.items():
            print(f"  {fname}: {content[:300]}...")

        # building context for LLM
        context_parts = []
        context_parts.append("* Full context:\n")
        context_parts.append("** Retrieved lore:\n")
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"lore {i+1}: {doc}\n")
        context_parts.append("** Referenced files:\n")
        for fname, content in linked_chunks.items():
            context_parts.append(f"**** Contents of attached file {fname}:\n{content.strip()}\n")
        context_parts.append(f"* Original Question: {original_question}\n")


        full_context = "".join(context_parts)

        return full_context
