from loguru import logger

from app.dto.get_suggestions import Property

import re, string, math, json
import pandas as pd
import numpy as np
#from collections import Counter

from sklearn.neighbors import BallTree

'''
from unidecode import unidecode

import spacy
from spacy_download import load_spacy

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize
# download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
'''

from app.dto.get_suggestions import GetSuggestionsRequest, GetSuggestionsResponse


class GetSuggestionsService:
    
    def __init__(self):
        # model to check the nature of words in French (pronouns, adjectives, etc.)
        #self.nlp = load_spacy("fr_core_news_sm")
        #self.stop_words = set(stopwords.words("french"))
        pass

    
    def get_suggestions(self, request: GetSuggestionsRequest) -> list:
        df_properties_user_pref = pd.read_json(json.dumps(request.properties_user_preferences, default=lambda o: o.__dict__, indent=4, ensure_ascii=False))
        df_properties_by_cluster = pd.read_json(json.dumps(request.properties_by_cluster, default=lambda o: o.__dict__, indent=4, ensure_ascii=False))
        df_properties_by_cluster = df_properties_by_cluster[~df_properties_by_cluster['ref'].isin(df_properties_user_pref['ref'])]
        nbr_similar_property = request.nbr_similar_property
        df_similar_properties = pd.DataFrame()
        iteration_suggestion = 0
        
        if len(df_properties_user_pref) >=nbr_similar_property :
            k_neighbors = 1
        else:
            k_neighbors = round(nbr_similar_property / len(df_properties_user_pref))
        
        dict_ball_tree_model = self.create_ball_trees_model(df_properties_by_cluster)
        
        
        while len(df_similar_properties) < nbr_similar_property and iteration_suggestion < 3:
            df_similar_properties = pd.DataFrame()
            for index, row in df_properties_user_pref.iterrows():
                ball_tree_model = dict_ball_tree_model[row['code_departement']][row['cluster']]
                df_similar_properties = pd.concat([df_similar_properties, 
                                                   self.find_similar_properties(row, df_properties_by_cluster, k_neighbors, ball_tree_model)], axis=0)

        df_similar_properties = df_similar_properties.drop_duplicates()
        
        df_similar_properties = df_similar_properties.sample(nbr_similar_property)
        return [json.loads(property.to_json()) for index, property in df_similar_properties.iterrows()]
        
        #result = self.content_based_algorithm_description(properties_user_pref, df_similar_properties)
        
        #properties_to_suggest = [json.loads(property.to_json()) for index, property in result.iterrows()]
       
    def create_ball_trees_model(self, df_properties_by_cluster: pd.DataFrame)->dict:
        dict_ball_trees = {}
        for dep in df_properties_by_cluster['code_departement'].unique():
            df_properties_by_dep = df_properties_by_cluster[df_properties_by_cluster['code_departement'] == dep]
            
            if dep not in dict_ball_trees.keys():
                dict_ball_trees[dep] = {}
            for cluster in df_properties_by_dep['cluster'].unique():
                if cluster not in dict_ball_trees[dep].keys():
                    df_cluster = df_properties_by_cluster[(df_properties_by_cluster['cluster'] == cluster) 
                                                           & (df_properties_by_cluster['code_departement'] == dep)]
                    
                    df_coords = df_cluster['coords'].str.split(';', expand=True)

                    # train our Ball Tree model with the coordinates of the properties in the cluster
                    balltree = BallTree(df_coords.values, leaf_size=40)
                    
                    dict_ball_trees[dep][cluster] = balltree
        return dict_ball_trees
                                      
    def find_similar_properties(self, property: pd.Series, df_properties_by_cluster: pd.DataFrame, k_neighbors: int, 
                                ball_tree_model: BallTree) -> pd.DataFrame:
        
        df_cluster = df_properties_by_cluster[(df_properties_by_cluster['cluster'] == property['cluster']) 
                                              & (df_properties_by_cluster['code_departement'] == property['code_departement'])]
        
        # retrieve the k_neighbors+1 properties closest to the targeted property (k_neighbors+1 because we remove the targeted property from the list)
        dist, indexes = ball_tree_model.query([[float(element) for element in property['coords'].split(";")]], k=k_neighbors+1)

        return df_cluster.iloc[indexes[0]]
    
    # the functions below represent the second (optional) step of
    # the recommendation algorithm which uses the asset descriptions to perform a content-based algorithm
    '''
    def clean_text_lem(self, text: str) -> list:
        if text != None:
            # remove special characters and punctuations
            text = re.sub(r'[%s]' % re.escape(string.punctuation.replace('-', '')), ' ', text)
            # remove numbers
            text = re.sub(r'\d+', '', text)
            # remove the doubled space
            text = re.sub(r'\s{2,}', ' ', text)
            # remove whitespaces at the beginning and the end
            text = text.strip()

            # keep compound words (like 'Sacré-Coeur' for example)
            compound_words = []
            no_compound_words = []

            for word in text.split():
                if '-' in word:
                    compound_words.append(word)
                else:
                    no_compound_words.append(word)

                text = " ".join(no_compound_words)

            # POS tagging with Spacy
            doc = self.nlp(text)
            pos_tags = [token.pos_ for token in doc]
            filtered_tokens = [token.text for token in doc if token.pos_ != "PROPN" and token.head.pos_ != "PROPN"]
            
            # exclude PRON and CCONJ
            text = [token for token, pos_tag in zip(word_tokenize(text), pos_tags) if pos_tag not in ["DET", "PRON", "CCONJ"]]
            text.extend(compound_words)
            text = " ".join(text)
            
            # We remove accents
            text = unidecode(text)
            # We tokenize data
            tokens = word_tokenize(text)
            # We lower the tokens
            tokens = [word.lower() for word in tokens]
            # We remove stopwords
            tokens = [word for word in tokens if not word in self.stop_words]
            # Lemmatize
            lemma = WordNetLemmatizer()
            tokens = [lemma.lemmatize(word, pos="v") for word in tokens if len(word) > 1]
            tokens = [lemma.lemmatize(word, pos="n") for word in tokens if len(word) > 1]

            return tokens

      
    def content_based_algorithm_description(self, df_preferences_user: pd.DataFrame, df_nearest_properties:pd.DataFrame) -> pd.DataFrame:
        df_preferences_user.loc[:, 'token_description'] = df_preferences_user['description'].apply(lambda x: self.clean_text_lem(x))
        df_nearest_properties.loc[:, 'token_description'] = df_nearest_properties['description'].apply(lambda x: self.clean_text_lem(x))

        words_properties_user = []
        occurrence_freq = []

        for word_tokens in df_preferences_user['token_description']:
            words_properties_user.extend(word_tokens)

        words_occurences = Counter(words_properties_user)
        size_words = len(words_properties_user)

        words_properties_user = []

        for word, count in words_occurences.items():
            words_properties_user.append(word)
            occurrence_freq.append(count / size_words)
        
        matrix = [[0] * len(words_properties_user) for _ in range(len(df_nearest_properties))]
        
        for i in range(len(df_nearest_properties)):
            property = df_nearest_properties.iloc[i]
            size_words_token = len(property['token_description'])
            for j in range(len(words_properties_user)):
                if words_properties_user[j] in property['token_description']:
                    matrix[i][j] = property['token_description'].count(words_properties_user[j]) / size_words_token

        cos_sim_dict = {}

        for i in range(len(matrix)):
            numerator = 0
            denominator_user = 0
            denominator_suggestions = 0
            index = 0
            for value in matrix[i]:
                if value != 0 :
                    numerator += value * occurrence_freq[index]
                    denominator_user +=  occurrence_freq[index] ** 2
                    denominator_suggestions += value ** 2
                index+=1
            if (math.sqrt(denominator_user) * math.sqrt(denominator_suggestions)) > 0:
                cos_sim_dict[i] = (numerator / (math.sqrt(denominator_user) * math.sqrt(denominator_suggestions)))
            else : 
                cos_sim_dict[i] = 0

        property_to_recommend = []

        sorted_cos_sim_dict = dict(sorted(cos_sim_dict.items(), key=lambda item: item[1], reverse=True))

        property_to_recommend = pd.DataFrame([df_nearest_properties.iloc[k] for k in sorted_cos_sim_dict.keys()])

        return property_to_recommend.drop(['token_description'], axis=1)
    '''