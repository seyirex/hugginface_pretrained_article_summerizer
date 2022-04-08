# Core Pkgs
import streamlit as st
from function import *
# EDA Pkgs
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Utils
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    menu = ["Home","Storage","About"]
    create_table()
    
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.title("Demo")
        
        st.sidebar.subheader("Tuning/Settings")
        # max_length= st.sidebar.slider("Maximum length of the generated text ",30,100)
        # top_k= st.sidebar.slider(" limits the sampled tokens to the top k values ",1,100)
        # temperature= st.sidebar.slider("Controls the craziness of the text ",0.7,100.0)
        model_type = st.sidebar.selectbox("Model type", options=["Bart","T5"])
        
        upload_doc = st.file_uploader("Upload a .txt, .pdf, .docx file for summarization")
        
        st.markdown("<h3 style='text-align: center; color: red;'>OR</h3>",unsafe_allow_html=True,)

        plain_text = st.text_area("Type your Message...",height=200)

        if upload_doc:
            clean_text = preprocess_plain_text(extract_text_from_file(upload_doc))
        else:
            clean_text = preprocess_plain_text(plain_text)
            
        summarize = st.button("Summarize...")        
        
        # called on toggle button [summarize]
        if summarize:
            if model_type == "Bart":
                text_to_summarize = clean_text

                with st.spinner(
                    text="Loading Bart Model and Extracting summary. This might take a few seconds depending on the length of your text..."):
                    summarizer_model = bart()
                    summarized_text = summarizer_model(text_to_summarize, max_length=100, min_length=30)
                    summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
                    st.success("Data Submitted for model retraining")
                    postdate = datetime.now()
                    # Add Data To Database
                    add_data(text_to_summarize,summarized_text,postdate)
            
            elif model_type == "T5":
                text_to_summarize = clean_text

                with st.spinner(
                    text="Loading T5 Model and Extracting summary. This might take a few seconds depending on the length of your text..."):
                    summarizer_model = t5()
                    summarized_text = summarizer_model(text_to_summarize, max_length=100, min_length=30)
                    summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text]) 
                    st.success("Data Submitted for model retraining")
                    postdate = datetime.now()
                    # Add Data To Database
                    add_data(text_to_summarize,summarized_text,postdate)

            # else:
            #     text_to_summarize = clean_text

            #     with st.spinner(
            #         text="Loading Pegasus Model and Extracting summary. This might take a few seconds depending on the length of your text..."):
            #         summarizer_model = pegasus()
            #         summarized_text = summarizer_model(text_to_summarize, max_length=100, min_length=30)
            #         # summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text]) 
            #         st.success("Data Submitted for model retraining")
            #         postdate = datetime.now()
            #         # Add Data To Database
            #         # add_data(text_to_summarize,summarized_text,postdate)      
            
            res_col1 ,res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Generated Text Visualization")
                # Create and generate a word cloud image:
                wordcloud = WordCloud().generate(summarized_text)
                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()
                summary_downloader(summarized_text)   
                
            with res_col2:
                st.subheader("Summarized Text Output")
                st.success("Summarized Text")
                st.write(summarized_text)
    
    elif choice == "Storage":
        st.title("Manage & Monitor Results")
        # stored_data =  view_all_data() 
        # new_df = pd.DataFrame(stored_data,columns=["text_to_summarize","summarized_text","postdate"])
        # st.dataframe(new_df)
        # new_df['postdate'] = pd.to_datetime(new_df['postdate'])
   
    
    else:
        st.subheader("About")
        # html_temp ="""<div>
        #          <p></p>
        #          <p></p>
        #          </div>"""
        # st.markdown(html_temp, unsafe_allow_html=True)
        


if __name__ == '__main__':
	main()