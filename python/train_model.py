from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import torch

def main():

    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map="auto",
                                             trust_remote_code=False, 
                                             revision="gptq-4bit-32g-actorder_True")


if __name__ == "__main__":
    main()
