# Voting Assistance Chatbot

This repository contains the code for a Voting Assistance Chatbot designed to simplify access to voting-related information. Built with **LangChain** and **Chainlit**, the chatbot uses advanced natural language processing (NLP) techniques to assist users with queries about voter registration, eligibility, polling stations, and other voting processes.

## Features

- **Natural Language Processing**: Users can interact in plain language to receive answers to voting-related questions.
- **Knowledge Retrieval**: Uses a `VectorStoreRetriever` to provide precise and relevant responses.
- **Asynchronous Processing**: Ensures fast and efficient handling of user queries without blocking operations.
- **User-Friendly Interface**: Built with **Chainlit** for a seamless and interactive user experience.
- **Modular Architecture**: Easily extensible to include more features, such as multilingual support or additional knowledge sources.

## Technology Stack

- **Python**: Backend logic and chatbot implementation.
- **LangChain**: For creating conversational AI workflows and retrieval-based question answering.
- **Chainlit**: For building the user interface and chatbot interaction framework.
- **VectorStore**: Stores and retrieves knowledge base documents for accurate answers.

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/voting-assistance-chatbot.git
   cd voting-assistance-chatbot



## Model: LLaMA 2 7B Chat

This project uses the `llama-2-7b-chat.ggmlv3.q8_0.bin` model for generating responses. You can download this model from **Hugging Face** by following these instructions:

### Steps to Download the Model:

1. **Sign up on Hugging Face** (if you don’t already have an account):
   - Visit [Hugging Face](https://huggingface.co/) and create an account.
   - Log in to your account.

2. **Accept the LLaMA 2 License**:
   - Visit the official LLaMA 2 page on Hugging Face: [Meta’s LLaMA 2 Models](https://huggingface.co/meta-llama).
   - Select the `llama-2-7b-chat.ggmlv3.q8_0.bin` model.
   - Click "Access Repository" and agree to the license terms.

3. **Download the Model**:
   - Use `git-lfs` to clone the model repository:
     ```bash
     git lfs install
     git clone https://huggingface.co/<repo-name-for-llama-2-7b-chat>
     ```
   - Navigate to the downloaded directory to find the `llama-2-7b-chat.ggmlv3.q8_0.bin` file.

4. **Move the Model File**:
   - Place the downloaded model file in your project's `models/` directory (or any directory referenced in your code):
     ```bash
     mkdir -p models
     mv path_to_downloaded_file/llama-2-7b-chat.ggmlv3.q8_0.bin models/
     ```

### Requirements

Make sure you have the following Python libraries installed:
- **`transformers`**: For model loading and inference.
- **`langchain`**: To integrate the model into your chatbot pipeline.

Install them using:
```bash
pip install transformers langchain
