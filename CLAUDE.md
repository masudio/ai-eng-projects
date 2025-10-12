# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI Engineering Projects repository containing 5 hands-on projects completed over 5 weeks, plus a capstone project. The projects build capabilities with large language models, retrieval, tool use, research workflows, and multimodality.

## Project Structure

- Each project is contained in its own directory (`project_1/`, `project_2/`, etc.)
- Projects are primarily Jupyter notebooks (`.ipynb` files) designed for educational purposes
- The main README.md contains comprehensive setup instructions and project descriptions
- Currently only Project 1 is available (LLM Playground)

## Development Environment

### Running Options
1. **Google Colab** (recommended for projects 1 and 5):
   - Upload notebook to Colab
   - Install required packages from `requirements.txt` in first cell
   - Add API tokens using `os.environ[...] = "value"`
   - Adjust file paths for Colab environment

2. **Local with Conda** (recommended for projects 2, 3, and 4):
   - Each project includes `environment.yml` file with dependencies
   - Create environment: `conda env create -f environment.yml`
   - Activate environment: `conda activate <ENV_NAME>`
   - Launch Jupyter: `jupyter notebook`

### Common Dependencies
Projects typically use these libraries:
- PyTorch (`torch`) for deep learning
- Transformers (`transformers`) for LLMs
- Hugging Face Hub (`huggingface_hub`) for models and datasets
- LangChain (`langchain`) for LLM applications
- OpenAI API (`openai`) for GPT models
- Tokenization (`tiktoken`) for OpenAI models
- NumPy (`numpy`) for numerical computing
- Streamlit or Gradio for UI components

## API Keys
Projects are designed to work without specific API keys by default, but may require:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Claude models
- `GOOGLE_API_KEY` for Gemini models
- `HUGGINGFACEHUB_API_TOKEN` for HF models
- `TAVILY_API_KEY` or `SERPAPI_API_KEY` for web search
- `PINECONE_API_KEY` for vector stores

## Weekly Project Topics
1. **Project 1**: LLM Playground (tokenization, GPT-2, decoding strategies)
2. **Project 2**: Customer-Support Chatbot (RAG, embeddings, Streamlit)
3. **Project 3**: Ask-the-Web Agent (tool calling, function schemas, web search)
4. **Project 4**: Deep Research System (reasoning workflows, multi-agent systems)
5. **Project 5**: Multimodal Agent (text-to-image, text-to-video, Gradio UI)
6. **Week 6**: Capstone Project (custom system design)

## Development Notes
- Projects are designed flexibly - multiple implementation approaches are encouraged
- Code sections marked with "your code here" require implementation
- No formal submission required - focus on learning and experimentation
- Models are loaded from Hugging Face Hub and cached locally after first download