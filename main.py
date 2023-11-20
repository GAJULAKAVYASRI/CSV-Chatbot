import streamlit as st
import tempfile
import pandas as pd
import openai
import matplotlib.pyplot as plt
from pandasai import SmartDataframe

# Set your OpenAI API key here
openai.api_key = "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"

# Function to load the LLM model
def load_llm():
    return openai

# Function to perform statistical analysis and generate plots
def analyze_and_plot_data(df, plot_type, x_column, y_column):
    st.subheader("Statistical Analysis:")

    # Perform basic statistical analysis
    st.write("Mean:\n", df.mean())
    st.write("Median:\n", df.median())
    st.write("Mode:\n", df.mode().iloc[0])
    st.write("Standard Deviation:\n", df.std())
    st.write("Correlation Matrix:\n", df.corr())

    st.subheader("Plots:")

    # Generate the plot based on user input or prompt the user for details
    if plot_type == "Histogram":
        column_to_plot = st.selectbox("Select column for histogram:", df.columns)
        fig, ax = plt.subplots()
        ax.hist(df[column_to_plot])
        st.pyplot(fig)
        return fig, ax

    elif plot_type == "Scatter Plot" or plot_type == "Line Plot":
        if x_column is None or y_column is None:
            st.warning("Please select X-axis and Y-axis columns.")
        else:
            fig, ax = plt.subplots()
            ax.scatter(df[x_column], df[y_column]) if plot_type == "Scatter Plot" else ax.plot(df[x_column], df[y_column])
            st.pyplot(fig)
            return fig, ax

# Function to handle questions related to statistics and plotting
def handle_questions(df, question):
    # Check if the question is related to plotting
    if "plot" in question.lower() and "between" in question.lower():
        # Extract column names from the question
        words = question.split()
        index_of_between = words.index("between")
        column1 = words[index_of_between + 1]
        column2 = words[index_of_between + 3]

        # Check if the columns exist in the dataframe
        if column1 in df.columns and column2 in df.columns:
            # Ask the user for plot type
            plot_type = st.selectbox("Select plot type:", ["Histogram", "Scatter Plot", "Line Plot"])
            analyze_and_plot_data(df, plot_type, column1, column2)

    # Check if the question is related to the mean of a specific column
    elif "mean" in question.lower() and "of" in question.lower():
        # Extract column name from the question
        words = question.split()
        index_of_of = words.index("of")
        column_to_mean = words[index_of_of + 1]

        # Check if the column exists in the dataframe
        if column_to_mean in df.columns:
            # Calculate and display the mean of the specified column
            st.write(f"Mean of {column_to_mean}: {df[column_to_mean].mean()}")

    # Check if the question is related to the average of a specific column
    elif "average" in question.lower() and "of" in question.lower():
        # Extract column name from the question
        words = question.split()
        index_of_of = words.index("of")
        column_to_average = words[index_of_of + 1]

        # Check if the column exists in the dataframe
        if column_to_average in df.columns:
            # Calculate and display the mean of the specified column
            st.write(f"Average of {column_to_average}: {df[column_to_average].mean()}")

    # Check if the question is related to the correlation of two columns
    elif "correlation" in question.lower() and "of" in question.lower():
        # Extract column names from the question
        words = question.split()
        index_of_of = words.index("of")
        column1 = words[index_of_of + 1]
        column2 = words[index_of_of + 3]

        # Check if the columns exist in the dataframe
        if column1 in df.columns and column2 in df.columns:
            # Calculate and display the correlation between the specified columns
            st.write(f"Correlation between {column1} and {column2}: {df[column1].corr(df[column2])}")

# Streamlit UI
st.title("Chat with CSV using Llama2 🦜")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>CSV-Openai❤️ </a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    df = pd.read_csv(tmp_file_path)
    df_ai = SmartDataframe(df, config={"verbose": True})

    # Perform statistical analysis and generate plots
    fig, ax = analyze_and_plot_data(df_ai, "Histogram", None, None)

    # Chat functionality
    st.subheader("Chat:")
    user_input = st.text_input("Query:", placeholder="Ask me anything about the data")
    if st.button("Send"):
        # Handle questions related to statistics and plotting
        handle_questions(df_ai, user_input)

        # Use OpenAI for general chat
        response = load_llm().Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=150
        )
        st.write(response.choices[0].text)


import streamlit as st
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

st.title("Chat with CSV using Llama2 🦙🦜")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]  # :wave:

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)

            if "plot" in user_input.lower():
                plot_data(data, user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

def plot_data(data, user_input):
    # Parse the user's query to extract plot details
    # Example: "Plot histogram of column_name" or "Plot scatter plot of column1 vs column2"
    # Extracted details will be used to generate the appropriate plot

    # Use pandasai to handle plotting based on user's query
    df = pd.DataFrame(data)
    llm = OpenAI()
    df = SmartDataframe(df, config={"llm": llm, "verbose": True})
    response = df.chat(user_input)

    # Check if the response contains plot details
    if response and "plot" in response.lower():
        st.write(response)  # Display the response to the user
        st.pyplot(df.plot())  # Use pandas plotting function
