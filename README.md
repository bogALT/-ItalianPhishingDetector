# Italian Phishing Detector

This is a thesis project developed by Bojan Poletan at [DISI @ University of Trento](https://www.disi.unitn.it/it) in collaboration with the [Cybersecurity department of Fondazione Bruno Kessler (FBK)](https://www.fbk.eu/it/cybersecurity/), Trento, Italy.

### Description
This project investigates the performance of Encoder-only architectures—specifically **BERT** and **RoBERTa**—in detecting Italian-language phishing threats. Utilizing a curated dataset of over 8,000 native Italian emails and an anonymization pipeline, the research demonstrates that fine-tuning models on localized, native data (Italian in this case) significantly outperforms models trained on English-centric datasets. The results highlight the necessity of language-specific training for capturing the nuanced semantic patterns of regional  cyber threats, this may also be applied to enterprises that may train the model to identify threats based on their services and communications.

----------

### Notes
To ensure seamless reproducibility and eliminate environment-specific dependency conflicts, this project has been migrated to **Kaggle**. By leveraging a containerized cloud environment, users can execute the entire pipeline immediately without manual configuration, guaranteeing consistent results across different platforms. The code is also included in this repository.

### Files
The project is so composed: it has the datasets and the python code to run on Kaggle

 - **eng-vs-ita-phishing-detector** is a python file that contains the code that can be run on Kaggle.
 - **fine-tuning-ita.zip** contains the italian fine-tuning datase (contact Matteo Rizzi Mrizzi@fbk.eu) 

Personal information have been replaced with placeholders, for example: NAME_PLACEHOLD in place of a real person's name, USERNAME_PLACEHOLD in place of a username and so on.
