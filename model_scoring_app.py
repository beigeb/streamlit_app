import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from lime.lime_tabular import LimeTabularExplainer
import openai

# OpenAI API key
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Set the page layout
st.set_page_config(layout="wide")

# Path to your logo image
logo_path = 'tubi-logo-78AAA023DB-seeklogo.com.png'  # Replace with your actual logo path

# Display logo image
st.image(logo_path, width=70)  # Adjust width as needed

# Custom CSS to position the logo at the top left corner
# Custom CSS to position the logo at the top left corner
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("{logo_path}");
            background-repeat: no-repeat;
            background-position: 20px 10px;  /* Adjust positioning from top and left */
            background-size: 30px;  /* Adjust size of the logo */
            padding-left: 60px;  /* Adjust spacing from the left */
            padding-top: 20px;   /* Adjust spacing from the top */
        }}
    </style>
    """, 
    unsafe_allow_html=True
)

# Load the SentenceTransformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the saved OLS model
@st.cache_resource
def load_ols_model():
    return joblib.load('ols_model.pkl')

# Load PCA models
@st.cache_resource
def load_pca_model(path):
    return joblib.load(path)

# Function to apply PCA and expand columns
def apply_pca_and_expand(df, column_name, pca_model_path):
    embeddings = np.stack(df[column_name].values)
    pca = load_pca_model(pca_model_path)
    components = pca.transform(embeddings)
    component_columns = [f"{column_name}_PCA_{i+1}" for i in range(components.shape[1])]
    component_df = pd.DataFrame(components, columns=component_columns, index=df.index)
    df = pd.concat([df, component_df], axis=1)
    return df

# Function to explain a single prediction and return an explanation
def explain_prediction(instance, model, explainer):
    exp = explainer.explain_instance(instance, model.predict)
    return exp

# Function to interpret LIME explanations
def get_text_interpretation(lime_explanation):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Interpret the following LIME explanation in a more detailed and human-readable way and keep the explanation limited to 50 words or less: {lime_explanation}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# Function to preprocess and score data
def preprocess_and_score(df_score):
    df_score['actor'] = df_score['actor'].fillna('').astype(str)
    df_score['actress'] = df_score['actress'].fillna('').astype(str)
    df_score['genres'] = df_score['genres'].fillna('').astype(str)
    df_score['director'] = df_score['director'].fillna('').astype(str)

    model = load_embedding_model()
    df_score['Actor_Embedding'] = df_score['actor'].apply(lambda x: model.encode(x))
    df_score['Genre_Encoding'] = df_score['genres'].apply(lambda x: model.encode(x))
    df_score['Actress_Encoding'] = df_score['actress'].apply(lambda x: model.encode(x))
    df_score['Director_Encoding'] = df_score['director'].apply(lambda x: model.encode(x))

    columns_to_drop = [
        'tconst', 'actor', 'actress', 'archive_footage', 'archive_sound',
        'casting_director', 'cinematographer', 'composer', 'director', 'editor',
        'producer', 'production_designer', 'self', 'writer', 'titleId', 'ordering',
        'title', 'region', 'language', 'types', 'attributes', 'titleType', 
        'primaryTitle', 'originalTitle', 'genres'
    ]
    df_score2 = df_score.drop(columns=columns_to_drop)

    df_score2 = apply_pca_and_expand(df_score2, 'Genre_Encoding', 'pca_genre_model.pkl')
    df_score2 = apply_pca_and_expand(df_score2, 'Actor_Embedding', 'pca_actor_model.pkl')
    df_score2 = apply_pca_and_expand(df_score2, 'Actress_Encoding', 'pca_actress_model.pkl')
    df_score2 = apply_pca_and_expand(df_score2, 'Director_Encoding', 'pca_director_model.pkl')

    df_score2.drop(columns=['Genre_Encoding', 'Actor_Embedding', 'Actress_Encoding', 'Director_Encoding'], inplace=True)

    ols_model = load_ols_model()
    df_score2['rating'] = ols_model.predict(df_score2)

    features = df_score2.drop(columns=['rating']).values
    feature_names = df_score2.drop(columns=['rating']).columns

    explainer = LimeTabularExplainer(training_data=features, feature_names=feature_names, mode='regression')

    results = []
    for i in range(len(df_score2)):
        instance = features[i]
        exp = explain_prediction(instance, ols_model, explainer)
        primary_title = df_score.iloc[i]['primaryTitle']
        results.append({
            'primaryTitle': primary_title,
            'rating': df_score2.iloc[i]['rating'],
            'lime_explanation': exp.as_list()
        })

    results_df = pd.DataFrame(results)
    results_df['text_interpretation'] = results_df['lime_explanation'].apply(get_text_interpretation)

    return results_df

st.markdown(
    """
    <style>
        .title-text {
            color: violet;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Streamlit UI
st.title(':violet[Platform Content Fitment Estimator]')
st.write("Upload Your Excel file to get the predictions and explanations.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df_score = pd.read_excel(uploaded_file)
    st.write("File uploaded successfully!")
    
    results_df = preprocess_and_score(df_score)
    
    st.write("Predictions:")
    st.dataframe(results_df[['primaryTitle', 'rating']])

    st.write("Detailed Explanations:")

    # Apply custom CSS for wrapping text
    st.markdown("""
    <style>
        /* Ensure text wraps in both header and data cells */
        .dataframe th, .dataframe td {
            white-space: normal !important;
            word-wrap: break-word;
            max-width: 300px; /* Adjust maximum width as needed */
        }
    </style>
    """, unsafe_allow_html=True)

    for index, row in results_df.iterrows():
        st.write(f"**{row['primaryTitle']}**: {row['text_interpretation']}")
        st.write("--------------------------------------------------------------------------------------------------------------------------------------------------------------")

    csv = results_df.to_csv(index=False)
    st.download_button(label="Download Predictions as CSV", data=csv, file_name='results_df.csv', mime='text/csv')
