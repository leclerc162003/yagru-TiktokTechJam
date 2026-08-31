# Shop Shop: Your Favourite Shopping Companion - Get What You Need ASAP!

![alt text](image.png)

Shop Shop is a fully offline conversational agent that is built for E-Commerce Shopping. It understands user's messages, learns the intent behind those messages, and continues to clarify and ask for more information to identify the user's ideal product. 

Shop Shop only uses the Python Standard Library and SQLite FTS5. It does not require any LLM, API or Network. This means it is fully offline, which allows it to run quickly and efficiently.

## Setup and installation

### Requirements

- Python 3.10 or newer
- A Python SQLite build with FTS5 

### Reproduce the results

### 1. Clone and enter the repository

```bash
git clone <https://github.com/leclerc162003/yagru-TiktokTechJam.git>
cd yagru-TiktokTechJam
```

### 2. Run the tests

```bash
python3 -m evaluator.local_evaluator
```


## Limitations and Reflection


1. **Limited Coverage** 
    
    Shop Shop is optimised for the dataset provided for this project and its specific categories. Therefore, the results may not transfer to other datasets. 

    If we had more time, we would expand it to be more adaptable to other categories of products and make it more robust in terms of cross-category intent and attribute understanding. 
  

2. **Intent Understanding Limitations** 
    
    Without the use of LLMs, unusual phrasings, typos and intent may be easily misunderstood. We have tried to ensure our Shop Shop is able to tackle these issues by having long vocabulary lists and more, but there are still limitations to these measures. 

    If we had more time and tokens, a well implemented LLM could have been used to better understand the intent of the user's message and to learn the user's preferences to recommend better products. 

3. **Lack of New User Information** 
    
    If the user is a new user, Shop Shop would have no learned preference to rely on when there is insufficient information provided by the user, which may cause it to produce less accurate results. 

    If we had more time, and if it was within the scope, other data related to the new user (from onboarding preferences questions (questions new users are asked when they first opened the app), data from the new user behaviour, data from cross platforms) would allow shop shop to have a better understanding of the user to give a more accurate result.


## Team Member Contributions

1. **Chua Jun Hong** 
  - Implemented Session State 
  - Implemented Reranking Feature
  - Implemented Intent-Overriding
  - Implemented Hybrid Retrieval
  - Implemented Signature Pool Feature


2. **Lau Jia Wen** 
  - Implemented Preferences feature 
  - Implemented Dual-Track Routing
  - Implemented Persistent Interaction Memory
  - Implemented Store and Learn User Preferences Feature
  - Implemented Self-Evolution Feature
