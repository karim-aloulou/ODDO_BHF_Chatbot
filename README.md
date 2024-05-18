# Solution overview:
Create a financial advisor chatbot for Oddo BHF's clients by providing the required functionalities, integrating the CSR strategy, while guaranteeing the scalability and reliability of this chatbot.
Data is highly confidential and is put  private and not accessible.


## What do you find in this project ?
1) Used Technologies:
   
   Langchain as a Workflow Framework, for the general pipeline and RAFT (RAG+RAFT) purposes.

   Mistral 7B as an LLM.

   Azure for computational ressources and work environment.
   
   Huggingface for storing private data.
   
   Gradio for Deployment.
   
3) Data:
   
   2665 pdfs were ingested plus it’s Metadata.
   
4) Functionalities:
   
   Chat history and context history feature.
   
   Snippet provision.
   
   Image extraction feature.
   
5) Pipelines:
   
   General architecture:
   
   ![pipeline](./static/pipeline.png)
   
   Main Pipeline:
   ![Main_pipeline](./static/Main_pipeline.png)

   Data ingestion Pipeline:
   ![Data_ingestion_pipeline](./static/Data_ingestion_pipeline.png)

   Data Preprocessing for Finetuning Pipeline:
   ![Data_ingestion_for_finetuning](./static/Data_ingestion_for_finetuning_pipeline.png)

   Finetuning Pipeline:
   ![Finetuning](./static/Finetuning_pipeline.png)



## Credits

 Credits for ODDO-BHF for the data.
 
 Credits for the project structure and implementation goes to INNOVISION
 
 INNOVISION Team Members: 
 - Med Karim Akkari
 - Karim Aloulou
 - Yosr Abbassi
 - Nadia Bedhiafi
 - Med Hedi Souissi
 - Med Dhia Mediouni
 - Sarra Gharsallah



