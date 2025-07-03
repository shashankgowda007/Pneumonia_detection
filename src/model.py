from transformers import pipeline
import torch
import warnings

class RAGModel:
    def __init__(self):
        self.model_type = "simple"
        try:
            # First try to load the full RAG model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.set_default_dtype(torch.float16)
                from transformers import RagTokenizer, RagSequenceForGeneration
                self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
                self.model = RagSequenceForGeneration.from_pretrained(
                    "facebook/rag-token-base",
                    torch_dtype=torch.float16
                )
                self.model = self.model.eval()
                self.model_type = "rag"
                
        except Exception:
            # Fallback to simpler model if RAG fails
            try:
                self.model = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_type = "qa"
            except Exception as e:
                raise RuntimeError(f"Failed to initialize any model: {str(e)}")

    def generate_answer(self, question, document_text=None):
        try:
            if self.model_type == "rag":
                with torch.no_grad():
                    if document_text:
                        # Use document as context for RAG
                        inputs = self.tokenizer(
                            question,
                            document_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )
                    else:
                        # General knowledge mode
                        inputs = self.tokenizer(question, return_tensors="pt")
                    
                    generated_ids = self.model.generate(
                        inputs["input_ids"],
                        max_length=200,
                        num_beams=2
                    )
                    answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Check if answer seems relevant
                    if document_text and not any(word.lower() in answers[0].lower() 
                                              for word in document_text.split()[:20]):
                        return ["I don't have enough information from the document to answer this."]
                    return answers
                    
            elif self.model_type == "qa":
                context = document_text if document_text else """
                General knowledge covers topics like:
                - Geography, History, Science
                - Literature and Mathematics
                """
                result = self.model(question=question, context=context)
                if result["score"] < 0.1:
                    return ["I couldn't find a good answer to that question."]
                return [result["answer"]]
            else:
                return ["System is initializing - please try again shortly"]
        except Exception as e:
            return [f"Sorry, I encountered an error: {str(e)}"]
