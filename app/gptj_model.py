# app/gpt_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoConfig
import torch
import os
import requests
import urllib.parse
from urllib.parse import unquote
import re
import json



class GPTJModel:
    def __init__(self):
        # === CloudFront URL ===
        cloudfront_url = "https://d3501cutc1ugpi.cloudfront.net/ai_app/Gen+AI/"
        model_prefix = "gpt-neo-125M/"
        offload_prefix = "offload/"

        
        # === Local temp paths ===
        model_path = "/tmp/gpt-neo-125M"
        offload_path = "/tmp/offload"

        # Create directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(offload_path, exist_ok=True)

        # === Download from CloudFront ===
        self._download_cloudfront_folder(cloudfront_url, model_prefix, model_path)
        self._download_cloudfront_folder(cloudfront_url, offload_prefix, offload_path)
        
        # Ensure config.json exists and has correct model_type
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            # Create default GPT-Neo config
            config = GPTNeoConfig(
                vocab_size=50257,
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                max_position_embeddings=2048,
                use_cache=True,
                bos_token_id=50256,
                eos_token_id=50256,
                attention_types=[[["global", "local"], 6]],
                window_size=256,
                rotary=True,
                rotary_dim=64,
                activation_function="gelu_new",
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                scale_attn_weights=True,
                use_parallel_residual=True,
            )
            config.save_pretrained(model_path)
        else:
            # Update existing config if needed
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            if 'model_type' not in config_data:
                config_data['model_type'] = 'gpt_neo'
            if 'attention_types' not in config_data:
                config_data['attention_types'] = [[["global", "local"], 6]]
            if 'num_layers' not in config_data:
                config_data['num_layers'] = 12
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

        print(f"Loading model from {model_path}")
        
        # Check for tokenizer files
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
        missing_files = [f for f in tokenizer_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            print(f"Missing tokenizer files: {missing_files}")
            # Initialize tokenizer with default GPT-Neo configuration
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neo-125M",
                padding_side="left",
                truncation_side="left"
            )
            # Save tokenizer files
            self.tokenizer.save_pretrained(model_path)
        else:
            # Initialize tokenizer with local files
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                truncation_side="left"
            )
        
        # Ensure we have a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Check for model weights
        model_files = ['pytorch_model.bin', 'model.safetensors']
        has_model_weights = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
        
        if not has_model_weights:
            print("Model weights not found, downloading from Hugging Face...")
            # Download and save the model
            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-neo-125M",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            model.save_pretrained(model_path)
            self.model = model
        else:
            # Load model with explicit configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                offload_folder=offload_path
            )
        
        # Move model to device if not using device_map
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
            
        print("Model loaded successfully")
       
    # def generate_answer(self, question: str):
    #     inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
    #     outputs = self.model.generate(inputs["input_ids"], max_length=100)
    #     return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _download_cloudfront_folder(self, base_url, prefix, local_path):
        """Download files from CloudFront URL to local directory."""
        print(f"Downloading from CloudFront URL {base_url}, prefix {prefix}")
        os.makedirs(local_path, exist_ok=True)

        # List of files to download
        files_to_download = [
            'config.json',
            'pytorch_model.bin',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json',
            'merges.txt'
        ]

        for file_name in files_to_download:
            file_url = f"{base_url}{prefix}{file_name}"
            dest_path = os.path.join(local_path, file_name)
            
            try:
                print(f"Downloading {file_url} to {dest_path}")
                response = requests.get(file_url, stream=True)
                response.raise_for_status()
                
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                print(f"Error downloading {file_name}: {str(e)}")
                continue

    def generate_answer(self, question: str):
        if not question or not isinstance(question, str):
            return "Please provide a valid question."
            
        # Create a more structured and explicit prompt with context
        prompt = f"""You are an AI assistant providing clear and accurate explanations. 
                    Please provide a concise and well-structured answer to the following question.
                    Focus on providing factual information and avoid repetition.

                    Question: {question}

                    Answer:"""

        try:
            # Tokenize input with padding and attention mask
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Increased input length
            ).to(self.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate output with more controlled parameters
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,  # Using max_new_tokens instead of max_length
                num_return_sequences=1,
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.7,       # Lower top_p for more focused sampling
                top_k=20,        # Lower top_k for more focused sampling
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=2.0,  # Increased to strongly discourage repetition
                no_repeat_ngram_size=5,  # Increased to prevent repeated phrases
                early_stopping=True,
                length_penalty=1.5  # Added to encourage more concise responses
            )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and clean the answer
            if "Answer:" in decoded:
                answer = decoded.split("Answer:")[-1].strip()
            else:
                answer = decoded.strip()
            
            # Post-process the answer
            answer = self._clean_response(answer)
                
            # Validate the answer
            if not answer or len(answer) < 10:
                return "I apologize, but I couldn't generate a proper response. Please try again."
                
            print(f"Generated answer: {answer}")
            return answer

        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

    def _clean_response(self, text: str) -> str:
        """Clean and format the response text."""
        # Remove any trailing incomplete sentences
        if '.' in text:
            text = text[:text.rindex('.') + 1]
            
        # Remove any repeated phrases
        words = text.split()
        unique_words = []
        for word in words:
            if len(unique_words) < 2 or word != unique_words[-1] or word != unique_words[-2]:
                unique_words.append(word)
                
        return ' '.join(unique_words)


gptj = GPTJModel()


