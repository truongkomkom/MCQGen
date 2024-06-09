import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

#Loading json file
with open("Response.json" , 'r') as file:
    response_json_str = file.read()  # Đọc toàn bộ nội dung của file
    RESPONSE_JSON = json.loads(response_json_str)

#creating a title for the app
st.title("MCQs Creator Application with LangChain 🦜")

#Create a form using st.form
with st.form ("user_inputs"):
    #File Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

    #Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    #Input Fields
    subject = st.text_input("Insert Subject", max_chars=20)

    #Quiz Tone
    tone=st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple" )

    #Add Button
    button=st.form_submit_button("Create MCQs")

    #Check if the button is clicked and all fields havev input

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading"):
            try:
                text=read_file(uploaded_file)
                #Count tokens and the cost of API call
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        })

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict): #Check respone is a dict
                    quiz=response.get("quiz", None) #Get "quiz", if don't have quiz is None
                    if quiz is not None:
                        start_idx = quiz.find('{')
                        # Extract the substring starting from the first '{'
                        json_str = quiz[start_idx:]
                        table_data=get_table_data(json_str)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)

                            #Display the review in a text box as well
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
