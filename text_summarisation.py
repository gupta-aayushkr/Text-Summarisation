import re
import nltk
import streamlit as st
import heapq
from nltk.cluster.util import cosine_distance
import networkx as nx
import numpy as np
import scipy as sp
nltk.download('punkt')
## article
st.title("NTLK: Text Summarisation ⛳️")

article_text = st.text_area("Enter the text:", height=200)
Sentence_lines = st.number_input("Summarised Sentence Lines", value=1)
if st.button("Summarise"):
    pass

# article_text="Just what is agility in the context of software engineering work? Ivar Jacobson [Jac02a] provides a useful discussion: Agility  has become today’s buzzword when describing a modern software process. Everyone is agile. An agile team is a nimble team able to appropriately respond to changes. Change is what software development is very much about. Changes in the software being built, changes to the team members, changes because of new technology, changes of all kinds that may have an impact on the product they build or the project that creates the product. Support for changes should be built-in everything we do in software, something we embrace because it is the heart and soul of software. An agile team recognizes that software is developed by individuals working in teams and that the skills of these people, their ability to collaborate is at the core for the success of the project.In Jacobson’s view, the pervasiveness of change is the primary driver for agility. Software engineers must be quick on their feet if they are to accommodate the rapid changes that Jacobson describes.  But agility is more than an effective response to change. It also encompasses the philosophy espoused in the manifesto noted at the beginning of this chapter. It encourages team structures and attitudes that make communication (among team members, between technologists and business people, between software engineers and their managers) more facile. It emphasizes rapid delivery of operational software and deemphasizes the importance of intermediate work products (not always a good thing); it adopts the customer as a part of the development team and works to eliminate the “us and them” attitude that continues to pervade many software projects; it recognizes that planning in an uncertain world has its limits and that a project plan must be ﬂ exible.  Agility can be applied to any software process. However, to accomplish this, it is essential that the process be designed in a way that allows the project team to adapt tasks and to streamline them, conduct planning in a way that understands the ﬂ uidity of an agile development approach, eliminate all but the most essential work products and keep them lean, and emphasize an incremental delivery strategy that gets working software to the customer as rapidly as feasible for the product type and operational environment. "
if article_text:
    article_text = article_text.lower()

    #removing double spaces and not alphabet words
    clean_text = re.sub('[^a-zA-Z]', ' ', article_text)
    clean_text = re.sub('\s+',' ', clean_text)

    #tokenising article into separate sentence done by nltk
    sentence_list = nltk.sent_tokenize(article_text)

    # nltk.download('stopwords')
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    word_frequencies = {}
    for word in nltk.word_tokenize(clean_text):
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word]/maximum_frequency

    sentence_scores = {}

    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence):
            if word in word_frequencies and len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    summary = heapq.nlargest(Sentence_lines, sentence_scores, key=sentence_scores.get)

    result = " ".join(summary)
    st.subheader("📈 Frequency Method")
    st.write(result)

if article_text:
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    article_text = re.sub("\n",' ', article_text)
    def read_article():
        # filedata = file.readline()
        # print(filedata)
        article = article_text.split(". ")
        # print(article)
        sentences=[]
        for i in article:
            sentences.append(i.replace("[^a-zA-Z]"," ").split(" "))
            # sentences.pop()
        # print(sentences)
        return sentences


    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        all_words = list(set(sent1+sent2))
        
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] +=1
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] +=1
        return 1-cosine_distance(vector1, vector2)

    def gen_sim_matrix(sentences, stop_words):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                        continue
                similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2])
        return similarity_matrix

    def generate_summary(top_n=5):
        stop_words = stopwords
        summarize_text = []
        sentences = read_article()
        sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
        sentences_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
        scores = nx.pagerank(sentences_similarity_graph)
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
        st.subheader("🧮 Cosine Similarity Method")
        st.write(". ".join(summarize_text))

    generate_summary(Sentence_lines)
