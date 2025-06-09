# Databricks notebook source
# MAGIC %md
# MAGIC # install packages

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-sdk==0.49.0 "databricks-langchain>=0.4.0" databricks-agents mlflow[databricks] langchain==0.3.25 langchain_core==0.3.59 databricks-vectorsearch==0.55 pydantic==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# # Initialize the WorkspaceClient
# client = WorkspaceClient()

# # Get all catalogs
# catalogs = client.catalogs.list()

# # Extract and display catalog names
# catalog_names = [catalog.name for catalog in catalogs]
# print(catalog_names)

# COMMAND ----------

catalog = "workspace"
db = "default"
table = "transcriptionData"

# COMMAND ----------

# %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC # Building our Chain

# COMMAND ----------

import mlflow

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = 'patient_data'

# COMMAND ----------

rag_chain_config ={
    "databricks_resources": {
        "llm_endpoint_name": "databricks-llama-4-maverick",
        "vector_search_endpoint_name": 'virtualphysicalchatbot',
    },
    "input_example": {
        "messages": [{"content": "I am having a fever, could you help me with it?", "role": "user"}]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        # "llm_prompt_template": "You are a trusted AI assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know. Here is the history of the current conversation you are having with your user: {chat_history}. And here is some context which may or may not help you answer the following question: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template": "You are a trusted AI assistant that helps patient diagnose the reason for their illness and give suggestions on medications or procedures to help ease those symptoms. If you do not know the answer to a question, you truthfully say you do not know. Here is the history of the current conversation you are having with your user: {chat_history}. And here is some context which may or may not help you answer the following question: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "chat_history", "question"],
    },
    "retriever_config": {
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 5, "query_type": "ann"},
        "schema": {"chunk_text": "content", "document_uri": "url", "primary_key": "id"},
        "vector_search_index": "workspace.default.transcriptiondataindex_v2",
    },
}
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
# model_config = mlflow.models.ModelConfig(development_config=rag_chain_config)

# COMMAND ----------

# MAGIC %%writefile chain.py
# MAGIC import os
# MAGIC import mlflow
# MAGIC from operator import itemgetter
# MAGIC from databricks_langchain.chat_models import ChatDatabricks
# MAGIC from databricks_langchain.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import PromptTemplate
# MAGIC from langchain_core.runnables import RunnablePassthrough
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC def extract_previous_messages(chat_messages_array):
# MAGIC     messages = "\n"
# MAGIC     for msg in chat_messages_array[:-1]:
# MAGIC         messages += (msg["role"] + ": " + msg["content"] + "\n")
# MAGIC     return messages
# MAGIC
# MAGIC def combine_all_messages_for_vector_search(chat_messages_array):
# MAGIC     return extract_previous_messages(chat_messages_array) + extract_user_query_string(chat_messages_array)
# MAGIC
# MAGIC #Get the conf from the local conf file
# MAGIC model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC vector_search_schema = retriever_config.get("schema")
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     endpoint=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=retriever_config.get("vector_search_index"),
# MAGIC     columns=[
# MAGIC         vector_search_schema.get("id"),
# MAGIC         vector_search_schema.get("content"),
# MAGIC         # vector_search_schema.get("document_uri"),
# MAGIC     ],
# MAGIC ).as_retriever(search_kwargs=retriever_config.get("parameters"))
# MAGIC
# MAGIC # Required to:
# MAGIC # 1. Enable the RAG Studio Review App to properly display retrieved chunks
# MAGIC # 2. Enable evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=vector_search_schema.get("id"),
# MAGIC     text_column=vector_search_schema.get("content"),
# MAGIC     # doc_uri=vector_search_schema.get("document_uri")
# MAGIC )
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = PromptTemplate(
# MAGIC     template=llm_config.get("llm_prompt_template"),
# MAGIC     input_variables=llm_config.get("llm_prompt_template_variables"),
# MAGIC )
# MAGIC
# MAGIC # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(combine_all_messages_for_vector_search)
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC         "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages)
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)
# MAGIC
# MAGIC # COMMAND ----------
# MAGIC
# MAGIC # chain.invoke(model_config.get("input_example"))

# COMMAND ----------

from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
import os
# Log the model to MLflow
with mlflow.start_run(run_name="virtual_physicain_rag"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='rag_chain_config.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema
        resources=[
            DatabricksVectorSearchIndex(index_name=model_config.get("retriever_config").get("vector_search_index")),
            DatabricksServingEndpoint(endpoint_name=model_config.get("databricks_resources").get("llm_endpoint_name"))
        ],
        extra_pip_requirements=["databricks-connect"]
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

from databricks import agents
MODEL_NAME = "virtual_physicain_rag"
MODEL_NAME_FQN = f"{catalog}.{db}.{MODEL_NAME}"

# COMMAND ----------

instructions_to_reviewer = f"""### Instructions for Testing the our Databricks Documentation Chatbot assistant

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing our assistant. Your contributions are essential to delivering a high-quality product to our end users."""

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_NAME_FQN)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=MODEL_NAME_FQN, model_version=uc_registered_model_info.version, scale_to_zero=True)

# Add the user-facing instructions to the Review App
agents.set_review_instructions(MODEL_NAME_FQN, instructions_to_reviewer)

wait_for_model_serving_endpoint_to_be_ready(deployment_info.endpoint_name)

# COMMAND ----------

