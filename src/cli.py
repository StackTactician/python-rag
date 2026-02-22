
import sys
import argparse
from rag import query_rag

DEFAULT_MODEL = "openrouter/aurora-alpha"
FAST_MODELS = {
    "1": "openrouter/aurora-alpha"
}

def interactive_mode(model_name=DEFAULT_MODEL):
    print(f"RAG CLI - Standard Input Mode")
    print(f"Using Model: {model_name}")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            print("\nThinking...", end="", flush=True)
            
            # Print newline before streaming
            print("\rAnswer:   \n")
            
            sources = []
            for chunk, docs in query_rag(query, model_name=model_name, stream=True):
                if docs:
                    sources = docs
                print(chunk, end="", flush=True)
            
            print("\n\nSources:")
            for doc, score in sources:
                print(f"- {doc.metadata.get('source', 'Unknown')} (Score: {score:.4f})")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

def single_query(query, model_name=DEFAULT_MODEL):
    sources = []
    print(f"Using Model: {model_name}\n")
    for chunk, docs in query_rag(query, model_name=model_name, stream=True):
        if docs:
            sources = docs
        print(chunk, end="", flush=True)
    print() # Newline at end

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Command Line Interface")
    parser.add_argument("query", nargs="?", help="The query string (optional)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-m", "--model", help="Model name or shortcut", default=DEFAULT_MODEL)
    
    args = parser.parse_args()
    
    # Resolve model alias
    model = FAST_MODELS.get(args.model, args.model)

    if args.interactive:
        interactive_mode(model)
    elif args.query:
        single_query(args.query, model)
    else:
        interactive_mode(model)
