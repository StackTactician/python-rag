"""NanoRAG command-line interface for interactive and one-shot queries."""

import argparse

from src.rag import DEFAULT_MODEL, query_rag

FAST_MODELS = {
    "1": DEFAULT_MODEL,
}


def interactive_mode(model_name: str = DEFAULT_MODEL):
    print("NanoRAG CLI - Standard Input Mode")
    print(f"Using Model: {model_name}")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("-" * 50)

    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print("\nAnswer:\n")
            sources = []
            for chunk, docs in query_rag(query, model_name=model_name, stream=True):
                if docs:
                    sources = docs
                print(chunk, end="", flush=True)

            if sources:
                print("\n\nSources:")
                for doc, score in sources:
                    print(f"- {doc.metadata.get('source', 'Unknown')} (Score: {score:.4f})")
            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"\nError: {exc}")


def single_query(query: str, model_name: str = DEFAULT_MODEL):
    print(f"Using Model: {model_name}\n")
    for chunk, _docs in query_rag(query, model_name=model_name, stream=True):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoRAG Command Line Interface")
    parser.add_argument("query", nargs="?", help="The query string (optional)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-m", "--model", help="Model name or shortcut", default=DEFAULT_MODEL)

    args = parser.parse_args()
    model = FAST_MODELS.get(args.model, args.model)

    if args.interactive:
        interactive_mode(model)
    elif args.query:
        single_query(args.query, model)
    else:
        interactive_mode(model)
