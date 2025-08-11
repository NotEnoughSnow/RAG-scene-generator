from core.RAG import RAG as RAG

def main():

    retriever = RAG()

    query = "what's outside the tavern ????"

    # preprocess query

    # retrieve context
    full_context = retriever.answer_question(query)

    # generate image

    # display description + image

    print("\n#### Full Context Sent to LLM ####")
    print(full_context)

    print("\n#### Question ####")
    print(query)


if __name__ == "__main__":
    main()