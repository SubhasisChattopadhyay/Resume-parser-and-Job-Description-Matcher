#!/usr/bin/env python
# coding: utf-8

# # Loading the Dataset and Pre-processing the Data

# In[78]:


#importing the required libraries
import os
import pdfplumber
import pandas as pd
import numpy as np
import re


# In[79]:


# Directory containing the PDF resume files
resume_dir = 'C:\\Users\\SUBHASIS\\Downloads\\archive\\data\\data'


# In[80]:


# Function to clean text and remove unwanted characters
def clean_text(text):
    # Removing special characters and non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text


# # Extracting Details from the Data

# In[81]:


# Initializing lists to store extracted information
ids = []
categories = []
skills_list = []
education_list = []
experience_list = []


# In[82]:


# Recursively looping through subdirectories and PDF files
for root, dirs, files in os.walk(resume_dir):
    for filename in files:
        if filename.endswith('.pdf'):
            # Extracting ID from the filename and removing '.pdf' extension
            resume_id = os.path.splitext(filename)[0]
            ids.append(resume_id)

            # Initializing variables to store extracted information
            category = None
            skills = []
            education = []
            experience = []

            # Accessing the PDF file using pdfplumber
            with pdfplumber.open(os.path.join(root, filename)) as pdf:
                # Getting the text from the first page
                first_page = pdf.pages[0]
                text = first_page.extract_text()

                # Extracting the category from the first line since it's the heading of each Resume
                lines = text.split('\n')
                if len(lines) > 0:
                    category = lines[0].strip()

                # Initializing a variable to keep track of the current section
                current_section = None

                # Looping through each page in the PDF
                for page in pdf.pages:
                    text = page.extract_text()

                    # Cleaning the extracted text
                    text = clean_text(text)

                    # Splitting the text into lines
                    lines = text.split('\n')

                    # Looping through each line
                    for line in lines:
                        # Checking for section headings
                        if "Skills" in line:
                            current_section = "Skills"
                        elif "Education" in line:
                            current_section = "Education"
                        elif "Experience" in line:
                            current_section = "Experience"

                        # Checking if we're inside a section
                        elif current_section:
                            # Appending the line to the appropriate section
                            if current_section == "Skills":
                                skills.append(line)
                            elif current_section == "Education":
                                education.append(line)
                            elif current_section == "Experience":
                                experience.append(line)

            # Appending to lists
            categories.append(category)
            skills_list.append(skills)
            education_list.append(education)
            experience_list.append(experience)


# In[89]:


# Ensuring all lists have the same length by filling with empty strings
max_len = max(len(ids), len(categories), len(skills_list), len(education_list), len(experience_list))

ids += [''] * (max_len - len(ids))
categories += [''] * (max_len - len(categories))
skills_list += [''] * (max_len - len(skills_list))
education_list += [''] * (max_len - len(education_list))
experience_list += [''] * (max_len - len(experience_list))


# In[90]:


# Creating a DataFrame
data = {
    'ID': ids,
    'Category': categories,
    'Skills': skills_list,
    'Education': education_list,
    'Experience': experience_list
}

df = pd.DataFrame(data)
df.head()


# In[91]:


df.head(20)


# In[92]:


#loading the Job Description dataset from hugging face
from datasets import load_dataset
# Downloading the Hugging Face dataset
job_descriptions = load_dataset("jacob-hugging-face/job-descriptions")


# In[93]:


#converting job description dataset to a dataframe from a dictionary
job_descriptions_df= pd.DataFrame(job_descriptions['train'])

#Cleaning the extracted text
job_descriptions_df['Job Description'] = job_descriptions_df['Job Description'].apply(clean_text)

job_descriptions_df.head()


# # Implementing distilBERT 

# In[94]:


#importing required libraries
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics.pairwise import cosine_similarity


# In[95]:


# Initializing DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


# In[102]:


# Defining batch size for processing job descriptions
batch_size = 2

# Defining batch size for processing CVs
cv_batch_size = 4  

# Defining number of CVs to sample
num_cvs_to_sample = 8  

# Defining maximum sequence length
max_seq_length = 512

# Initializing lists to store embeddings
job_desc_embeddings = []
resume_embeddings = []

# Processing job descriptions in batches
for i in range(0, len(job_descriptions_df), batch_size):
    print(f"Processing job descriptions batch {i//batch_size + 1}/{len(job_descriptions_df) // batch_size}")
    batch_job_desc = job_descriptions_df['job_description'][i:i+batch_size].tolist()  # Convert to a list of strings
    inputs = tokenizer(batch_job_desc, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    
    # Setting the device to CPU
    inputs = {key: value.to(torch.device('cpu')) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    job_desc_batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    job_desc_embeddings.extend(job_desc_batch_embeddings)

print("Finished processing job descriptions")

# Tokenizing sampled CVs and convert them to embeddings
for i in range(0, len(df), cv_batch_size):
    print(f"Processing CV batch {i//cv_batch_size + 1}/{len(df) // cv_batch_size}")
    batch_cvs = df[['Category', 'Skills', 'Education', 'Experience']][i:i+cv_batch_size]

    inputs = tokenizer(list(batch_cvs), return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)

    # Setting the device to CPU
    inputs = {key: value.to(torch.device('cpu')) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cv_batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    resume_embeddings.extend(cv_batch_embeddings)

print("Finished processing CVs")


# # Matching Top 5 Resumes with Job descriptions

# In[106]:


# Converting PyTorch tensors to NumPy arrays
job_desc_embeddings = np.array([emb.cpu().numpy() for emb in job_desc_embeddings])
resume_embeddings = np.array([emb.cpu().numpy() for emb in resume_embeddings])

# Calculating cosine similarities between job descriptions and CVs
similarities = cosine_similarity(job_desc_embeddings, resume_embeddings)

# Displaying top 5 CVs for each job description
top_n = 5  

# Looping through job descriptions and find top 5 CVs
for i in range(len(job_desc_embeddings)):
    job_desc_emb = job_desc_embeddings[i]
    sim_scores = similarities[i]
    top_cv_indices = sim_scores.argsort()[::-1][:top_n]

    print(f"Job Description {i + 1}: {job_descriptions_df['position_title'][i]}")
    print("Top CVs:")
    
    for j, idx in enumerate(top_cv_indices):
        if idx < len(df):
            print(f"CV {j + 1}: Similarity Score {sim_scores[idx]}")
            print(df['ID'].iloc[idx])
        else:
            print(f"CV {j + 1}: Index {idx} out of bounds")


# In[ ]:




