# bbc_nlp_pipeline

HM Land Registry - BBC News Sub-Category Classification and NLP Tasks

Michael Gardiner - michaeljamesgardiner@gmail.com

Overview - This repository contains a solution to a NLP challenge using articles from the BBC News website. The goal is to classify high-level news categories into finer sub-categories, and extract named entities and event summaries through an LLM (Meta LLaMA via Groq)

---

Task 1 - Subcategory Classification

  Using topic_modelling.py, the following subcategories were generated via prompt-based topic modeling for each main BBC category:

    business: Mergers and Acquisitions, Economic Growth and Indicators, Corporate Finance and Governance, Industry Performance, Monetary Policy and Currency, Corporate Restructuring and Litigation, Global Trade and Finance
    
    entertainment: Film, Music, Theatre and Performance, Awards and Nominations, Celebrity News and Obituaries, Arts and Culture, Industry Trends and Business
    
    politics: Election Politics, Immigration and Asylum Policy, Economic Policy and Budget, Foreign Policy and International Relations, Law and Order, Government Accountability and Scandal, Social Policy and Welfare
    
    sport: Athletics, Football, Rugby, Tennis
    
    tech: Cybersecurity, Consumer Electronics and Mobile Technology, Internet and Online Services, Gaming Industry, Intellectual Property and Copyright, Emerging Technologies

  Each article is then mapped to one of the derived subcategories using topic_classification.py.


Task 2 - Named Entity Recognition (NER)

  Handled by entity_recognition.py, this step extracts names of individuals mentioned in the articles and identifies their professions (e.g., "David Beckham, Footballer", "Tony Blair, Politician").


Task 3 - April Event Extraction

  Also using entity_recognition.py, the script extracts only those events that occurred during the month of April and provides a summry of those events.

---

Dataset

  Source - BBC News dataset - http://mlg.ucd.ie/datasets/bbc.html
  
  Format - 2225 text files organized by top-level category (Business, Entertainment, Politics, Sport, Tech), 
  
  corresponding to articles from the BBC news website from 2004-2005

---
Outputs

  All script outputs are saved as either .json or .txt files within /outputs

---

How to Run

In windows powershell run the following commands:

git clone https://github.com/michaeljamesgardiner/bbc_nlp_pipeline.git

cd "" insert project directory

py -m venv venv

.\venv\Scripts\activate

pip install -e . 

pip install llama-index-llms-groq python-dotenv

Define the variables with the .env file

Functions can be called via the follwing commands:

python -m topic_modelling --input_dir "" --output_dir ""

python -m entity_recognition --input_dir "" --output_dir "" --mode {media_personalities,april_events}

  --mode media_personalities: Extracts named people and their professions

  --mode april_events: Extracts and summarizes events from April only

python -m subcategory_classification --input_dir "" --output_dir ""

input_dir - output .JSON or .txt file location 

output_dir - input directory containing category subfolders of .txt articles
