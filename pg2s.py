#Example must be a dict of list
examples = {
    "put a hot bread in the refrigerator": {
        'truth':[
    'Walk to the pencil located on the right side of the desk near the edge.',
    'Pick up the pencil.',
    'Turn to face the trash bin next to the desk on the left side.',
    'Walk to the trash bin.',
    'Put the pencil into the trash bin.'
                ],
        'predict':[
    'Walk to the pencil located on the right side of the desk near the edge.',
    'Pick up the pencil.',
    'Turn to face the trash bin next to the desk on the left side.',
    'Walk to the trash bin.',
    'Put the pencil into the trash bin.'
            ]
    }
}

#import models
import numpy as np
import gensim.downloader as api
import spacy
from sentence_transformers import SentenceTransformer

wv = api.load('word2vec-google-news-300') 
nlp = spacy.load("en_core_web_sm")  

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def framing(sentences):
    '''
        create actions from the sets
    '''
    actions_and_states = []
    for sentence in sentences:
        sentence = sentence.lower()
        doc = nlp(sentence)
        state_i = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                state_i.append(token.text)
            elif token.pos_ == "NOUN" and token.dep_ in ["pobj", "dobj"]:
                if token.head.dep_ == "prep":
                    if token.text in state_i:
                        state_i[state_i.index(token.text)].replace(token.text, "")
                    else:
                        state_i.append(token.text)
        if state_i:
            actions_and_states.append(state_i)
    return actions_and_states


def goal_wise_similarity(truth, pred):
    '''
        Taken in input two sets of actions give the result of the similarity 
    '''
    truth_list = truth.copy()
    pred_list = pred.copy()
    conf = []
    for element in truth_list:
        verb = element[0]
        nouns = element[1:]
        best = None
        max_similarity = 0
        for element2 in pred_list:
            verb2 = element2[0]
            nouns2 = element2[1:]
            verbs_similarity = wv.similarity(verb, verb2)
            nouns_similarity = []
            min_len = min(len(nouns), len(nouns2))
            if len(nouns) != len(nouns2) and min_len == 0:
                nouns_similarity = 0
            else:
                if len(nouns) < len(nouns2):
                    nouns_similarity = [1 if wv.similarity(nouns[i], nouns2[i]) >= 0.708 else 0 for i in range(min_len)]
                else:
                    nouns_similarity = [1 if wv.similarity(nouns2[i], nouns[i]) >= 0.708 else 0 for i in range(min_len)]
                if not nouns_similarity:
                    nouns_similarity = 1
                else:
                    nouns_similarity = np.mean(nouns_similarity)
            if verbs_similarity * nouns_similarity > max_similarity and verbs_similarity > 0.708:
                max_similarity = verbs_similarity * nouns_similarity
                best = element2
        if best is not None:
            index_delete = pred_list.index(best)
            pred_list.pop(index_delete)
            conf.append(1)
        else:
            conf.append(0)
    if not conf:
        return 0
    return np.mean(conf)

def sentence_similarity(list1,list2):
    '''
        Taken in input two sets of sentences give the similarity between the plans
    '''
    truth_copy = list1.copy()
    pred_copy = list2.copy()
    conf = []
    for i in range(len(truth_copy)):
        max = float("-inf")
        best = None
        embeddings = model.encode(truth_copy[i])
        for j in range(len(pred_copy)):
            embeddings2 = model.encode(pred_copy[j])
            similarity = np.dot(embeddings,embeddings2)/(np.linalg.norm(embeddings)*np.linalg.norm(embeddings2))
            if similarity > max and similarity > 0.675:
                max = similarity
                best = pred_copy[j]

        if best is not None:
            pred_copy.remove(best)
            conf.append(1)
        else:
            conf.append(0)
    if len(conf) == 0:
        return 0
    return np.mean(conf)

truth = examples["put a hot bread in the refrigerator"]["truth"]
predict = examples["put a hot bread in the refrigerator"]["predict"]
truth_actions = framing(truth)
predict_actions = framing(predict)
goal_wise = goal_wise_similarity(truth_actions,predict_actions)
sentence_wise = sentence_similarity(truth,predict)

pg2s = goal_wise *0.5 + sentence_wise *0.5
print("PG2S : ", pg2s)