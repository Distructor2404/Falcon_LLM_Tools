from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import PyPDF2
import docx

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

if "code" not in st.session_state:
           st.session_state.code = False
if "textsum" not in st.session_state:
           st.session_state.textsum = False
if "language" not in st.session_state:
        st.session_state.language =False
if "sentiment" not in st.session_state:
        st.session_state.sentiment =False
if "email" not in st.session_state:
        st.session_state.email =False
if "question" not in st.session_state:
        st.session_state.question =False
if "chatwithcsv" not in st.session_state:
        st.session_state.chatwithcsv =False
if "maths" not in st.session_state:
        st.session_state.maths =False

if "final" not in st.session_state:
        st.session_state.final =False

if "after" not in st.session_state:
        st.session_state.after =True

if "hugkey" not in st.session_state:
     st.session_state.hugkey = ""



repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
keyy = st.session_state.hugkey





    




def falcon(question,keyy):
    

    from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token = keyy , model_kwargs={"temperature": 0.2, "max_new_tokens": 500}
        )

    template = """Answer the {question} """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question} "
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


    chain = LLMChain(llm=falcon_llm, prompt=chat_prompt)
    result = chain.run(question=question)
    return result

def falcon_text(questions,keyy):
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token = keyy , model_kwargs={"temperature": 0.2, "max_new_tokens": 2000}
        )
    

    template = """Question: {question}

Answer: Go through the Question and generate a concise summary for it."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)
    response = llm_chain.run(questions)
    
    return response


def falcon_senti(questions,keyy):
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token = keyy , model_kwargs={"temperature": 0.2, "max_new_tokens": 2000}
        )
    

    template = """Question: {question}

Answer: Please analyze the sentiment of the following statement."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)
    response = llm_chain.run(questions)
    
    return response

def falcon_trans(question,transfrom,transto,keyy):
    from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token = keyy , model_kwargs={"temperature": 0.2, "max_new_tokens": 2000}
        )

    template = """question : {question}
 You are a concise translation assistant. Translate the question from language {transfrom} to language {transto}. 
 """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


    chain = LLMChain(llm=falcon_llm, prompt=chat_prompt)
    result = chain.run(transfrom=transfrom, transto=transto, question=question)
    return result


def falcon_email(name,to,sub,mail,keyy):
    from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token = keyy , model_kwargs={"temperature": 0.2, "max_new_tokens": 2000}
        )

    template = """You are an email Generator,

    provide the output :
    from : {name} ,
    to: {to} ,
    subject: {sub} ,
    content: provide more details to sender in more than 50 words as much as possible targeting {mail} 
    end with thank you"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Provide the output in the standard email format."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


    chain = LLMChain(llm=falcon_llm, prompt=chat_prompt)
    result = chain.run(name = name,to=to, sub=sub, mail=mail)
    return result



def clear_chat_1():
            st.session_state.messages = [{"role": "assistant", "content": "Ask me anything😁"}]

def clear_chat_2():
            st.session_state.messages = [{"role": "assistant", "content": "Hello Coder😎!! How can I help you?"}]



def falcon_code(questions,keyy):

    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token = keyy , model_kwargs={"temperature": 0.2, "max_new_tokens": 2000}
        )
    

    template = """Question: {question}

Answer: Provide the required codes for the statement. Only provide the answer in coding basis"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)
    response = llm_chain.run(questions)
    
    print(response)
    return response

st.set_page_config(page_title="Falcon LLM")
with open("style.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)


with st.sidebar:
   keyyy = st.text_input("Enter your Huggingface Token")
   button_key = st.button("Enter Key")
   if(button_key):
        st.session_state.hugkey = keyyy
        st.session_state.final = True
        st.session_state.after = False        

        
        

   
        
   st.caption("Get your Token -> login to huggingface -> Account settings -> Access Tokens")
   st.caption("Your key will be not stored, it gets reset once you reload or close your page")
   
   if(st.session_state.final):

   
        st.subheader("Try all Different Usecases")

        question = st.button("Q/A Model")
        
        st.success("This section has no Fine-tuning, you can use any command to get whatever you want. This mode is best to test out the model as per your requirements.")
        st.write("")
        code = st.button("Code Generation")
        st.success("This section has been Fine-tuned to provide the best code solutions, you can ask any coding question regarding any language. This mode is best to test out the model in coding aspects.")
        st.write("")

        textsum = st.button("Text Summarizer")
        st.success("This model has been Fine-tuned to provide the best Summary from the enter paragraph or uploaded document.  ")
        st.write("")

        language = st.button("Language Translator")
        st.success("This model has been prompted efficiently, you can use this model to translate your text.")
        st.write("")

        sentiment = st.button("Sentiment Analysis")
        st.success("This Model will be analysing your emotions or sentiments from the text. You can test this to know more about the LLM regarding sentiments ")
        st.write("")

        email = st.button("Email Generator")
        st.success("This model will let you curate any email as per your requirement.")
        st.write("")

        csv = st.button("Chat with CSV")
        st.success("This model allow you to chat with your CSV and provide you the desired information.")
        st.write("")

        maths = st.button("Maths Problem Solver")
        st.success("Solve your maths problem using Falcon LLM")
        st.write("")


        st.caption("All the Above model is Fine-Tuned, the model is working on 7 Billion paramters. ")
        st.caption("Falcom LLM do not provide any specific model, all the use cases is generated by me.")

        if(question):
            clear_chat_1()
            st.session_state.question = True
            st.session_state.code = False
            st.session_state.textsum = False
            st.session_state.language = False
            st.session_state.sentiment = False
            st.session_state.email = False
            st.session_state.chatwithcsv = False
            st.session_state.maths = False


        if(code):
            clear_chat_2()
            st.session_state.question = False
            st.session_state.code = True
            st.session_state.textsum = False
            st.session_state.language = False
            st.session_state.sentiment = False
            st.session_state.email = False
            st.session_state.chatwithcsv = False
            st.session_state.maths = False

        if(textsum):
            st.session_state.question = False
            st.session_state.code = False
            st.session_state.textsum = True
            st.session_state.language = False
            st.session_state.sentiment = False
            st.session_state.email = False
            st.session_state.chatwithcsv = False
            st.session_state.maths = False

        if(language):
            st.session_state.question = False
            st.session_state.code = False
            st.session_state.textsum = False
            st.session_state.language = True
            st.session_state.sentiment = False
            st.session_state.email = False
            st.session_state.chatwithcsv = False
            st.session_state.maths = False

        if(sentiment):
            st.session_state.question = False
            st.session_state.code = False
            st.session_state.textsum = False
            st.session_state.language = False
            st.session_state.sentiment = True
            st.session_state.email = False
            st.session_state.chatwithcsv = False
            st.session_state.maths = False

        if(email):
            st.session_state.question = False
            st.session_state.code = False
            st.session_state.textsum = False
            st.session_state.language = False
            st.session_state.sentiment = False
            st.session_state.email = True
            st.session_state.chatwithcsv = False
            st.session_state.maths = False

        if(csv):
            st.session_state.question = False
            st.session_state.code = False
            st.session_state.textsum = False
            st.session_state.language = False
            st.session_state.sentiment = False
            st.session_state.email = False
            st.session_state.chatwithcsv = True
            st.session_state.maths = False

        if(maths):
            st.session_state.question = False
            st.session_state.code = False
            st.session_state.textsum = False
            st.session_state.language = False
            st.session_state.sentiment = False
            st.session_state.email = False
            st.session_state.chatwithcsv =  False
            st.session_state.maths = True

        


st.title("Falcon LLM")
if(st.session_state.after):
    st.caption("Enter your Hugging Face Token to access all the different model using FALCON LLM.")
    st.caption("Downloading the FALCON model in local is possible but to make it accessible for every one it needs to uploaded into a high speed server.")


   

if (st.session_state.question): 
   st.header("Question Answering ⁉️")

   if "messages" not in st.session_state:
      st.session_state.messages = [{"role": "assistant", "content": "Ask me anything😁"}]

   for message in st.session_state.messages:
      st.chat_message(message["role"]).markdown(message["content"])

   if prompt := st.chat_input("Search anything"):
               st.chat_message("user").markdown(prompt)
               st.session_state.messages.append({"role": "user", "content": prompt})
               with st.spinner("Thinking..."):
                  response=falcon(prompt,st.session_state.hugkey)
                  st.chat_message("assistant").markdown(response)
                  st.session_state.messages.append({"role": "assistant", "content": response})



if (st.session_state.code): 
   st.header("Code Generation ⚙️")

   if "messages" not in st.session_state:
      st.session_state.messages = [{"role": "assistant", "content": "Hello Coder😎!! How can I help you? "}]

   for message in st.session_state.messages:
      st.chat_message(message["role"]).markdown(message["content"])

   if prompt := st.chat_input("Hi Coder!! Search for the code you want"):
               st.chat_message("user").markdown(prompt)
               st.session_state.messages.append({"role": "user", "content": prompt})
               with st.spinner("Generating..."):
                  response=falcon_code(prompt,st.session_state.hugkey)
                  st.chat_message("assistant").markdown(response)
                  st.session_state.messages.append({"role": "assistant", "content": response})




if (st.session_state.textsum): 
   st.header("Text Summarization 🖊️")
   title3 = st.text_input('Enter you text')
   answer2 = "Go through the Question and generate a concise summary for it."
   generate = st.button("Generate Summary")

   uploaded_file = st.file_uploader("Upload any Document file to Summaries", type=["docx","pdf"])

   if uploaded_file is not None:
        # Read CSV file
        st.write("Uploaded file:", uploaded_file.name)

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = "Unsupported format"

        st.subheader("Extracted Text:")
        st.text_area("Extracted Text", text, height=400)
        generate21 = st.button("Generate Doc Summary")
        if(generate21):
            with st.spinner("Generating Summary...."):
                output = falcon_text(text,st.session_state.hugkey)
                st.write(output)


   if(generate):
      with st.spinner("Generating Summary...."):
            output = falcon_text(title3,st.session_state.hugkey)
            st.write(output)



if (st.session_state.language): 
   st.header("Language Translation 🎤")
   col3,col4 = st.columns(2)
   with col3:
      fromm = st.text_input('Translate from')

   with col4:
      langtrans = st.text_input('Translate to')
   title8 = st.text_input('Enter you text')
   generate2 = st.button("Translate")
   if(generate2):
      with st.spinner("Translating...."):
            output = falcon_trans(title8,fromm,langtrans,st.session_state.hugkey)
            st.write(output)

if (st.session_state.sentiment): 
      st.header("Sentiment Analysis 🧐")
      st.caption("Type any Sentence with emotions to test it out (For example: I recently watched a heartwarming movie, and it made me feel so happy and uplifted.)")
      st.caption(" Positive Statement : I had an incredible experience during my vacation to the beach. The sunsets were breathtaking, the ocean waves were soothing, and the company of my friends made it even more enjoyable. I felt so relaxed and rejuvenated throughout the trip.")
      st.caption("Negative Statement: The recent economic downturn has caused significant hardships for many families. With job losses and financial instability, people are facing tough times. It's disheartening to see the struggles people are going through, and I hope for a quick recovery for everyone affected.")

      st.caption("Neutral Statements: The conference covered a wide range of topics and had informative sessions. It was well-organized, and the speakers were knowledgeable. Overall, it was a satisfactory experience, and I appreciate the effort put into organizing the event.")
      title1 = st.text_input('Enter you text')
      generate = st.button("Get Your Sentiments")
      if(generate):
         with st.spinner("Analysing Sentiments...."):
            output = falcon_senti(title1,st.session_state.hugkey)
            st.write(output)

if (st.session_state.email): 
   st.header("Email Curator")
   col1,col2 = st.columns(2)
   with col1 :
        name = st.text_input("Your Name")

   with col2 :
        to = st.text_input('Enter Recipent salutation')

   sub = st.text_input('Enter Subject')
   mail = st.text_area('Enter something about your Email')
   generate = st.button("Curate Email")

   if(generate):
      with st.spinner("Generating Email...."):
            output = falcon_email(name,to,sub,mail,st.session_state.hugkey)
            st.write(output)
            
   

if (st.session_state.chatwithcsv): 
   st.header("Chat with CSV")
   st.write("")
   st.write("")

   st.write("")
   st.subheader("Coming Soon.....")


if (st.session_state.maths): 
   st.header("Maths Problem Solver")
   st.write("")
   st.write("")

   st.write("")
   st.subheader("Coming Soon.....")


            

   
        

      




