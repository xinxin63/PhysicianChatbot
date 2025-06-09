# PhysicianChatbot
Databricks hackathon 2025
%md
## Goal:
We are building a helpful AI chatbot that works as a physician which can help physicians or patients to diagnose based on the patients' symptoms and then provide the recommended approaches for the treatment.

## Conversation Example:

**Input**:
Patient Info : Age, Gender, Occupation, symptoms etc.,

**Model Activities**:
- Built Gen AI RAG app pipeline: \
    GenAI model: databricks-llama-4-maverick \
    llm RAG chain with question, context and chat history \
    Log the RAG app to Mlflow
- Model will get information around past diagnostics patterns
- Information around various treatment plans
- Suggest options for the treatment/next steps
- Model evaluation: 

For example : If the patient has backpain, It will suggest chiropractor sessions and recommendations on best way to sit and posture which can help subsiding the backpain

## Dataset
Disease Prevalence Rates -> Demographic
Physician Dictation Data : Family Medicine -> It has Office Visit transcripts from physicians across almost all US states. \
Source: https://marketplace.databricks.com/details/bc106eb9-e5dc-443c-b015-f7a9b6aa686f/Shaip_Physician-Dictation-Data-Family-Medicine \
All the datasets have been processed which included RecursiveCharacterTextSplitter and vector embeddings using GTEgte. Lastly them were saved in vector search index for online similarity search.

## Notebooks
Data preparation: Embedding Generation \
RAG app: virtual_physician_chatbot \
RAG chain config: rag_chain_config.yaml
RAG Chatbot UI: physicianchatbot_2025_06_09-21_19
