import os
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import glob
from web_search import search_sap_info, format_search_results

# Initialize colorama for colored output
init()

def load_knowledge_base():
    """Load all knowledge base files from the knowledge_base directory."""
    knowledge = []
    knowledge_files = glob.glob("knowledge_base/*.txt")
    
    if not knowledge_files:
        print(f"{Fore.YELLOW}Warning: No knowledge base files found in 'knowledge_base' directory.{Style.RESET_ALL}")
        return ""
    
    for file_path in sorted(knowledge_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                knowledge.append(content)
                print(f"{Fore.GREEN}Loaded knowledge from: {os.path.basename(file_path)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading {file_path}: {str(e)}{Style.RESET_ALL}")
    
    return "\n\n".join(knowledge)

def initialize_model():
    """Initialize the local model and tokenizer."""
    print(f"{Fore.YELLOW}Loading model and tokenizer... This might take a few minutes the first time.{Style.RESET_ALL}")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"{Fore.GREEN}Model loaded successfully!{Style.RESET_ALL}\n")
    return model, tokenizer

def generate_response(model, tokenizer, user_input, knowledge_base, conversation_history, max_length=200):
    """Generate a response from the model."""
    # Try to find relevant web information
    web_results = search_sap_info(user_input)
    web_info = format_search_results(web_results) if web_results else ""
    
    # Format the input with knowledge base context and web search results
    system_prompt = f"""You are an AI assistant specialized in SAP. Use the following knowledge base and web search results to help answer questions:

KNOWLEDGE BASE:
{knowledge_base}

WEB SEARCH RESULTS:
{web_info}

IMPORTANT INSTRUCTIONS:
1. First, use information from the knowledge base if available
2. If the knowledge base doesn't have the information:
   - Use relevant web search results if available
   - Clearly indicate when you're using web-sourced information
   - Be transparent about the source of information
3. If neither knowledge base nor web search has the information:
   - Clearly state that you don't have access to this information
   - Do not make assumptions or guesses
   - Suggest topics you can help with instead
4. Be precise and technical when discussing SAP topics
5. Explain SAP-specific terms when using them
6. Stay within your knowledge boundaries
7. Never make up information

Remember: Honesty about limitations is better than providing uncertain information."""
    
    # Format conversation history
    history_text = ""
    if conversation_history:
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in conversation_history])
        history_text = f"\nPrevious conversation:\n{history_text}\n"
    
    prompt = f"<|system|>\n{system_prompt}{history_text}\n<|user|>\n{user_input}\n<|assistant|>\nBased on the available information, "
    
    # Encode the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response.replace(prompt, "").strip()
    
    return response

def main():
    """Main chat loop."""
    print(f"{Fore.GREEN}Welcome to the SAP AI Assistant!")
    print(f"Loading knowledge base...{Style.RESET_ALL}\n")
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    
    # Initialize conversation history
    conversation_history = []
    
    try:
        # Initialize model and tokenizer
        model, tokenizer = initialize_model()
        
        while True:
            # Get user input
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                print(f"\n{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
                break
            
            try:
                # Generate and print AI response
                response = generate_response(model, tokenizer, user_input, knowledge_base, conversation_history)
                print(f"{Fore.GREEN}AI: {response}{Style.RESET_ALL}\n")
                
                # Add to conversation history (keep last 5 exchanges to manage context window)
                conversation_history.append((user_input, response))
                if len(conversation_history) > 5:
                    conversation_history.pop(0)
                
            except Exception as e:
                print(f"{Fore.RED}Error: An error occurred while getting the response:")
                print(f"{str(e)}{Style.RESET_ALL}\n")
    
    except Exception as e:
        print(f"{Fore.RED}Error: Could not initialize the model:")
        print(f"{str(e)}")
        print(f"Make sure you have enough disk space and RAM available.{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 