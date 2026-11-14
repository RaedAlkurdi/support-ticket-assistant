import json 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# loading articles
with open ('articles.json', 'r') as f:
    articles = json.load(f)

# loading  dev questions
with open ('dev_questions.json', 'r') as f:
    dev_questions = json.load(f)

#making a list of all text
texts = []
for article in articles:
    text = article['text']
    texts.append(text)

vecktorizer = TfidfVectorizer()
X = vecktorizer.fit_transform(texts)
hit = 0
answerable_count= 0

for i in range (len(dev_questions)):
    question = dev_questions[i]['question'] #choosing question 
    X2 = vecktorizer.transform([question])
    simlarity = cosine_similarity(X,X2)
    scores = simlarity[:, 0]
    ranked_indices = np.argsort(scores)[::-1]
    #print(f"Question {i+1}: {question}\nPredictions:\n")
    #for index in ranked_indices:
        #print(f'{articles[index]['id']}\n{articles[index]['title']}\n{scores[index]}\n')

    top3_ids = []
    for id in ranked_indices[:3]:
        id = articles[id]['id']
        top3_ids.append(id)

    expected_IDs = dev_questions[i]['relevant_article_ids']

    if dev_questions[i]["answerable"]:
        answerable_count += 1
        if any(expected_id in top3_ids for expected_id in expected_IDs):
            hit += 1

print(f"Hit@3: {hit/answerable_count}")

# 1. loading pretrained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. embeddings

embeddings = model.encode(texts)
print(embeddings.shape)
hit = 0
answerable_count= 0

# 3. applying everything again but for the model

for i in range (len(dev_questions)):
    question = dev_questions[i]['question'] #choosing question 
    X2_model = model.encode([question])
    simlarity = cosine_similarity(embeddings,X2_model)
    scores = simlarity[:, 0]
    ranked_indices = np.argsort(scores)[::-1]
    #rint(f"Question {i+1}: {question}\nPredictions:\n")
    #for index in ranked_indices:
        #print(f'{articles[index]['id']}\n{articles[index]['title']}\n{scores[index]}\n')


    top3_ids = []
    for id in ranked_indices[:3]:
        id = articles[id]['id']
        top3_ids.append(id)

    expected_IDs = dev_questions[i]['relevant_article_ids']

    if dev_questions[i]["answerable"]:
        answerable_count += 1
        if any(expected_id in top3_ids for expected_id in expected_IDs):
            hit += 1

print(f"Hit@3 for model: {hit/answerable_count}")

for i in range (len(articles)):
    tokens = model.tokenizer(articles[i]["text"], truncation=False)
    
    if len(tokens["input_ids"]) > model.max_seq_length:
        print (f'Article {i+1} excceds the limti')
    else:
        print(f'Article {i+1} is good')