# Databricks notebook source
# MAGIC %md
# MAGIC # Project Description

# COMMAND ----------

# MAGIC %md
# MAGIC ## Goal:
# MAGIC We are building a helpful AI chatbot that works as a physician which can help physicians or patients to diagnose based on the patients' symptoms and then provide the recommended approaches for the treatment.
# MAGIC
# MAGIC ## Conversation Example:
# MAGIC
# MAGIC **Input**:
# MAGIC Patient Info : Age, Gender, Occupation, symptoms etc.,
# MAGIC
# MAGIC **Model Activities**:
# MAGIC - Built Gen AI RAG app pipeline: \
# MAGIC     GenAI model: databricks-llama-4-maverick \
# MAGIC     llm RAG chain with question, context and chat history \
# MAGIC     Log the RAG app to Mlflow
# MAGIC - Model will get information around past diagnostics patterns
# MAGIC - Information around various treatment plans
# MAGIC - Suggest options for the treatment/next steps
# MAGIC - Model evaluation: 
# MAGIC
# MAGIC For example : If the patient has backpain, It will suggest chiropractor sessions and recommendations on best way to sit and posture which can help subsiding the backpain
# MAGIC
# MAGIC ## Dataset
# MAGIC Disease Prevalence Rates -> Demographic
# MAGIC Physician Dictation Data : Family Medicine -> It has Office Visit transcripts from physicians across almost all US states. \
# MAGIC Source: https://marketplace.databricks.com/details/bc106eb9-e5dc-443c-b015-f7a9b6aa686f/Shaip_Physician-Dictation-Data-Family-Medicine \
# MAGIC All the datasets have been processed which included RecursiveCharacterTextSplitter and vector embeddings using GTEgte. Lastly them were saved in vector search index for online similarity search.
# MAGIC
# MAGIC ## Notebooks
# MAGIC Data preparation: Embedding Generation \
# MAGIC RAG app: virtual_physician_chatbot \
# MAGIC RAG chain config: rag_chain_config.yaml
# MAGIC RAG chatbot UI: physicianchatbot_2025_06_09-21_19
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspiration
# MAGIC We have seen patients having difficulties understanding when to consult a physician or take home medication for their symptoms 
# MAGIC
# MAGIC ## What it does
# MAGIC **Input:** Patient Info : Age, Gender, Occupation, symptoms etc.,
# MAGIC
# MAGIC **Model Activities:**
# MAGIC
# MAGIC Model will get information around past diagnostics patterns
# MAGIC Information around various treatment plans
# MAGIC Suggest options for the treatment/next steps
# MAGIC Model evaluation:
# MAGIC For example : If the patient has backpain, It will suggest chiropractor sessions and recommendations on best way to sit and posture which can help subsiding the backpain
# MAGIC
# MAGIC ## How we built it
# MAGIC
# MAGIC Built Gen AI RAG app pipeline:
# MAGIC GenAI model: databricks-llama-4-maverick
# MAGIC llm RAG chain with question, context and chat history
# MAGIC Log the RAG app to Mlflow
# MAGIC
# MAGIC **Dataset used**
# MAGIC Disease Prevalence Rates -> Demographic Physician Dictation Data : Family Medicine -> It has Office Visit transcripts from physicians across almost all US states.
# MAGIC Source: https://marketplace.databricks.com/details/bc106eb9-e5dc-443c-b015-f7a9b6aa686f/Shaip_Physician-Dictation-Data-Family-Medicine
# MAGIC All the datasets have been processed which included RecursiveCharacterTextSplitter and vector embeddings using GTEgte. Lastly them were saved in vector search index for online similarity search.
# MAGIC
# MAGIC ## Challenges we ran into
# MAGIC
# MAGIC - Healthcare data on Databricks marketplace were not working and struggled to find correct dataset
# MAGIC - Integrating with vector index table created 
# MAGIC
# MAGIC ## Accomplishments that we're proud of
# MAGIC - The code runs successfully
# MAGIC
# MAGIC ## What we learned
# MAGIC
# MAGIC ## What's next for Physician chatbot
# MAGIC - Integrate with over the counter prescription details
# MAGIC - 

# COMMAND ----------

# MAGIC %md
# MAGIC
