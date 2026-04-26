import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(enc.encode(text))


def build_context(docs, max_tokens=3000):
    context = []
    current_tokens = 0

    for doc in docs:
        tokens = count_tokens(doc)

        if current_tokens + tokens > max_tokens:
            break

        context.append(doc)
        current_tokens += tokens

    return "\n".join(context)