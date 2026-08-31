# Shop Shop: Your Favourite Shopping Companion - Get What You Need ASAP!

![alt text](image.png)

Shop Shop is a fully offline coversational agent that is built for E-Commerce Shopping. It understands user's messages and it learns the intent behind those messages, it continues to clarify and ask for more information to identify the user's ideal product. 

Shop Shop only uses the Python Standard Library and SQLite FTS5. It does not require any LLM, API or Network. Hence, it is fully offline and this allows it to run faster and more efficiently.

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


## Limitations and reflection


1. **Limited Coverage** 
  Shop Shop is optimised for this specific category and dataset used in this project. Hence, the results may not transfer outside of this category/dataset. 

  If time was not a limitation, we would expand it to be more adaptable across other categories and make it more robust in terms of cross-category intent and attribute understanding. 
  

2. **Intent Understanding Limitations** 
  Without the use of LLMs, unusual phrasings, typos and intent may be misunderstood easily. We have tried to ensure our Shop Shop is robust enough to tackle these issues, but there are still limitations to these measures. 

  If time and cost was not a limitation, a well implemented LLM could have been used to understand and learn the intent of the user's message more to have a more accurate search. 

3. **Lack of New User Information** 
  If the user is a new user, the Shop Shop has no history or profile to reference with, hence it may produce an inaccurate result. 

  If time was not a limitation, and if it is within the scope, if there is avaliable data relating to the new user (from onboarding preferences questions (questions new users are asked when they first opened the app), data from the new user behaviour, data from cross platforms), this would allow shop shop to have a better understanding of the user and give a more accurate result.


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