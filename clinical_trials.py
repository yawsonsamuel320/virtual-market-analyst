# clinical_trial_app.py

import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from tqdm import tqdm
import pandas as pd

from openai import OpenAI
import openai
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

ONE_SHOT_EXAMPLAR = '''
Here is an example patient note, each sentence is led by a sentence_id:
0. Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower
extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain.
1. The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation.
2. Complicated by progressive lower extremity weakness and urinary retention.
3. Patient initially presented with RLE weakness where his right knee gave out with difficulty walking and right anterior
thigh numbness.
4. MRI showed a spinal cord conus mass which was biopsied and found to be anaplastic astrocytoma.
5. Therapy included field radiation t10-l1 followed by 11 cycles of temozolomide 7 days on and 7 days off.
6. This was followed by CPT-11 Weekly x4 with Avastin Q2 weeks/ 2 weeks rest and repeat cycle.
Here is an example clinical trial:
Title: Is the Severity of Urinary Disorders Related to Falls in People With Multiple Sclerosis
Target diseases: Fall, Multiple Sclerosis, Lower Urinary Tract Symptoms
Interventions: Clinical tests
Summary: Falls are a common problem in people with multiple sclerosis (PwMS) and can lead to severe
consequences (trauma, fear of falling, reduction of social activities). Prevention of falls is one of the priority targets
of rehabilitation for PwMS and walking difficulties, which can result of different factors (motor impairment, ataxia,
sensitive disorders, fatigability…). Urinary incontinence has been evoked as predictive of falls. But lower urinary
tract symptoms (LUTSs) are frequent in PwMS, the prevalence of LUTSs is high (32-96.8%) and increases with MS
duration and severity of neurological deficiencies and disabilities. Overactive bladder (OAB) is the most common
symptom. Despite its high prevalence and impact on quality of life, the severity of LUTSs has never been studied as
specific risk factor of falling. However, urinary urgency and urinary incontinence could lead to precipitation and thus
could increase the risk of falling in these patients.~The aim of the study was to assess the relationship between
severity of LUTSs and risk of falling in PwMS.~Patients were asked about the number of falls in the past three
months and in the past year, and the circumstances in which they occurred (frequency, home, outdoors, going to
void, during urinary urgency, nocturia). Severity of LUTSs were assessed by the Urinary Symptoms Profile (USP)
Score and patient were classified as with or without urinary incontinence. Number of micturition by night were
specifically asked. To take into account motor difficulties and fear of falling, other clinical evaluations were done.
The impact of MS on walking was assessed by the 12-Item Multiple Sclerosis Walking Scale (MSWS12)
questionnaire, the Expanded Disability Status Scale score, and by clinical test with the Time to be Ready to Void
(TRV). Fear of falling was assessed by a simple question and with Falls Efficacy Scale-International (FES-I)
Questionnaire.~The primary aim was to assess the relationship between severity of LUTSs and occurrence of falls
during the past 3 months. The primary outcome was the importance of overactive bladder (OAB) symptoms with
OAB USP score. The secondary outcomes were the existence of urinary incontinence, the warning time (defined as
the time from the first sensation of urgency to voiding or incontinence), the importance of nocturia and the other
scores of USP questionnaire (low stream and stress urinary incontinence).~The secondary aims were to look for the
relationship between severity of LUTSs and occurrence of falls during the past year, and to assess the relationship
between falls and the classical risk factors of falls.
Inclusion criteria: inclusion criteria:
age ≥ 18 years
Multiple sclerosis (MS) diagnosis
Lower urinary tract symptoms with or without treatment,
Expanded Disability Status Scale score between 1 and 6.5
Example plain JSON output without indent:
{"age \u2265 18 years": ["The patient is a 45-year-old man, older than 18.", [0], "included"], "Multiple sclerosis (MS)
diagnosis": ["The patient has never been diagnosed with multiple sclerosis, so he does not meet the target condition
of this trial.", [0, 1, 2, 3, 4, 5, 6], "not included"], "Lower urinary tract symptoms with or without treatment": ["The
patient has urinary retention complication and has been treated.", [0], "included"], "Expanded Disability Status Scale
score between 1 and 6.5": ["There is no relevant information of this score in the patient note.", [], "no relevant
information"]}
'''

# Function to load PDF and split into pages
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(pages)

    return splits


# Function to retrieve eligibility criteria from OpenAI model
def return_eligibility_criteria(criteria_type, eligibility_module):
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": '''Hello. You are a helpful assistant for clinical trial recruitment. 
            Give your answer and separate each item in lists with a new line character. If they are multiple groups/cohorts described, return the criteria for 
            only the group 1/cohort 1.
            Begin the list this way, Criteria: criteria 1, criteria 2...
            '''},
            {"role": "user", "content": f"{eligibility_module} What is the list of {criteria_type} for this trial?"},
        ]
    )

    return response.choices[0].message.content.replace('Criteria:', '') \
        .replace('\n1', '1')


# Function to retrieve clinical trials data
def get_clinical_trials(condition):
    base_url = "https://clinicaltrials.gov/api/query/full_studies"
    params = {
        "expr": condition,
        "max_rnk": 10,
        "fmt": "json"
    }
    st.write(f'**Condition:** {condition}')
    response = requests.get(base_url, params=params)
    trials_data = response.json().get("FullStudiesResponse", {}).get("FullStudies", [])
    return trials_data

# Function to process clinical trials data
def process_trials_data(trials_data):
    potential_trials = []
    
    progress_text = "Searching possible clinical trials..."
    my_bar = st.progress(0, text=progress_text) 
    
    # Loop through each study using streamlit progress bar
    for i, study in enumerate(trials_data):
        my_bar.progress(((i+1) / len(trials_data)), text=f'Searching possible clinical trials... {100*((i+1) / len(trials_data)): .2f}%')
        trial = {}
        nct_id = study['Study']['ProtocolSection']['IdentificationModule']['NCTId']
        title = study['Study']['ProtocolSection']['IdentificationModule'].get("BriefTitle", "")
        eligibility_module = study['Study']['ProtocolSection']['EligibilityModule']['EligibilityCriteria']
        inclusion_criteria = return_eligibility_criteria('inclusion criteria', eligibility_module)
        exclusion_criteria = return_eligibility_criteria('exclusion criteria', eligibility_module)
        inclusion_criteria_list = inclusion_criteria.replace('inclusion criteria', '').replace(':', '').replace('\n\n', '').split('\n')
        if '' in inclusion_criteria_list: 
            inclusion_criteria_list.remove('')
        exclusion_criteria_list = exclusion_criteria.replace(':', '').replace('\n\n', '').split('\n')
        if '' in exclusion_criteria_list: 
            exclusion_criteria_list.remove('')
        trial["NCT ID"] = nct_id
        trial["title"] = title
        trial["Inclusion Criteria"] = []
        trial["Exclusion Criteria"] = []
        for i, in_criterion in enumerate(inclusion_criteria_list):
            trial["Inclusion Criteria"].append(f"Inclusion Criterion {i+1}.: {in_criterion}")
        for i, ex_criterion in enumerate(exclusion_criteria_list):
            trial["Exclusion Criteria"].append(f"Exclusion Criterion {i+1}.: {ex_criterion}")
        potential_trials.append(trial)
    my_bar.empty()
    return potential_trials
chat_history = []


def generate_ranking_score(potential_trials, 
                           one_shot_examplar=ONE_SHOT_EXAMPLAR, 
                           db=None, 
                           chain=None, 
                           chat_history=chat_history):
    progress_text = "Ranking potential clinical trials..."
    my_bar = st.progress(0, text=progress_text) 
    for i, trial in enumerate(potential_trials):
        my_bar.progress(((i+1) / len(potential_trials)), text=f'Ranking potential clinical trials... {100*((i+1) / len(potential_trials)): .2f}%')
        inclusion_criteria_matching = []

        for inclusion_criterion in trial['Inclusion Criteria']:
            prompt = '''
                        Hello. You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient notes and diagnosis and one
                        inclusion criteria of a clinical trial to determine the patient's eligibility.
                        The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on
                        characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other
                        medical conditions.
                        You should check the inclusion criteria one-by-one, following the steps below:
                        1. For each inclusion criterion, first think step-by-step to explain if and how the patient note is relevant to the
                        criterion. You should explain in detail if there is relevant information.
                        2. Then, if there is relevant information, you must annotate a list of relevant sentence IDs of the patient note. If
                        there is no relevant information, you must annotate an empty list.
                        3. Finally, annotate the patient eligibility for this specific inclusion criterion: the eligibility must be 'no relevant
                        information' if there is no relevant information. Otherwise, the patient can only be 'included' or 'not included' if there
                        are relevant sentences. 'included' means that the patient meets the inclusion criterion, and 'not included' means
                        that the patient contradicts the inclusion criterion.
                        4. You should output only a JSON dict exactly formatted as: dict{str(inclusion_criterion):
                        list[str(relevance_explanation), list[int(sentence_id)], str('included'|'not included'|'no relevant information')]}
                    ''' + f'''
                        {one_shot_examplar}

                        Here is the clinical trial:
                        {inclusion_criterion}
                        Plain JSON output without indent:
                    '''

            # Get most likely sources where the question's answer is likely to be found
            matching_docs = db.similarity_search(prompt)
            # Compute the response from the documents
            answer = eval(chain.run(input_documents=matching_docs,
                                question=prompt))
            inclusion_criteria_matching.append(answer)
            chat_history.extend([HumanMessage(content=prompt), answer])
            exclusion_criteria_matching = []

        for exclusion_criterion in trial['Exclusion Criteria']:
            prompt = '''
            Hello. You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient notes and diagnosis and one
            exclusion criteria of a clinical trial to determine the patient's eligibility.
            The factors that allow someone to participate in a clinical study are called exclusion criteria. They are based on
            characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other
            medical conditions.
            You should check the exclusion criteria one-by-one, following the steps below:
            1. For each exclusion criterion, first think step-by-step to explain if and how the patient note is relevant to the
            criterion. You should explain in detail if there is relevant information.
            2. Then, if there is relevant information, you must annotate a list of relevant sentence IDs of the patient note. If
            there is no relevant information, you must annotate an empty list.
            3. Finally, annotate the patient eligibility for this specific exclusion criterion: the eligibility must be 'no relevant
            information' if there is no relevant information. Otherwise, the patient can only be 'excluded' or 'not excluded' if there
            are relevant sentences. 'excluded' means that the patient meets the exclusion criterion, and 'not excluded' means
            that the patient contradicts the exclusion criterion.
            4. You should output only a JSON dict exactly formatted as: dict{str(exclusion_criterion):
            list[str(relevance_explanation), list[int(sentence_id)], str('excluded'|'not excluded'|'no relevant information')]}
            ''' + f'''
            {one_shot_examplar}

            Here is the clinical trial:
            {exclusion_criterion}
            Plain JSON output without indent:
            '''

            # Get most likely sources where the question's answer is likely to be found
            matching_docs = db.similarity_search(prompt)
            # Compute the response from the documents
            answer = eval(chain.run(input_documents=matching_docs,
                                    question=prompt))
            
            exclusion_criteria_matching.append(answer)
            chat_history.extend([HumanMessage(content=prompt), answer])

        inclusion_pattern = [list(in_matches.values())[-1][-1] for in_matches in inclusion_criteria_matching]
        exclusion_pattern = [list(ex_matches.values())[-1][-1] for ex_matches in exclusion_criteria_matching]

        met_inclusion = inclusion_pattern.count('included')/len(inclusion_pattern)
        unmet_inclusion = inclusion_pattern.count('not included')/len(inclusion_pattern)
        met_exclusion = exclusion_pattern.count('excluded')/len(exclusion_pattern)
        unmet_exclusion = exclusion_pattern.count('not excluded')/len(exclusion_pattern)

        ranking_score = (met_inclusion - unmet_inclusion) - (met_exclusion - unmet_exclusion)

        trial['Ranking Score'] = '{:.1f} %'.format(ranking_score*100)
    
    return potential_trials
    
# Function to display the Streamlit app
def main():
    # Center 4th-IR Logo from './4th-ir-logo.png' to be small
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image('./4th-ir-logo.png', width=100)
    st.title("4th-IR Virtual Market Analyst")

    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_file:
        st.write("PDF file uploaded successfully!")
        
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()

        # Save the uploaded file to the temporary directory
        pdf_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split the PDF file
        splits = load_and_split_pdf(pdf_file_path)

        # Initialize OpenAI model and FAISS
        client = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = load_qa_chain(client, chain_type="stuff")
        db = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Automatically generate condition using OpenAI model
        condition_prompt = '''
        Hello. You are a helpful assistant for clinical trial recruitment. We need to get the condition that the patient is suffering from to place that into the clinical trials
        API. What condition is the patient suffering from? 
        Mention the main category of the condition and the main category of organ system it affects
        without any reference to the specific location of the organ. Return only the answer in about 1-4 words which I can place into the input field.
        The answer should be in this format {Condition} - {Main category of organ system}
        '''
        matching_docs = db.similarity_search(condition_prompt)
        # Show spinner while chain generates condition text
        with st.spinner('Extracting medical condition from patient notes...'):
            condition = chain.run(input_documents=matching_docs,
                               question=condition_prompt)
        st.success('Done!')

        # Retrieve clinical trials data
        with st.spinner('Retrieving clinical trials data...'):
            trials_data = get_clinical_trials(condition)

        if trials_data:
            # Process clinical trials data
            potential_trials = process_trials_data(trials_data)
        
            # Generate ranking score for potential trials
            ranked_potential_trials = generate_ranking_score(potential_trials, db=db, chain=chain)
            
            # Convert ranked_potential_trials to a DataFrame
            ranked_potential_trials = pd.DataFrame(ranked_potential_trials)
            
            # Create new column for url of clinical trial (eg: https://clinicaltrials.gov/study/{NCT ID})
            ranked_potential_trials['URL'] = ranked_potential_trials['NCT ID'].apply(lambda x: f"https://clinicaltrials.gov/study/{x}")
            
            # Sort out ranked_potential_trials by 'Ranking Score' (eg: '10%, 0.5%, -5%')
            ranked_potential_trials['Ranking Score'] = ranked_potential_trials['Ranking Score'].str.replace('%', '').astype(float)
            ranked_potential_trials = ranked_potential_trials.sort_values(by='Ranking Score', ascending=False)
            ranked_potential_trials['Ranking Score'] = ranked_potential_trials['Ranking Score'].astype(str) + ' %'

            # Display results in a table
            st.write("## Potential Clinical Trials")
            st.write("Here are the potential clinical trials for the given condition:")
            st.write(ranked_potential_trials)

        else:
            st.write("No clinical trials found for the given condition.")

if __name__ == "__main__":
    main()
