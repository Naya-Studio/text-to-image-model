### Imports

import os
import requests
import numpy as np
import pandas as pd
import random
import pickle
import sklearn
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import math

from pymongo import MongoClient
from flask import Flask, request

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.manifold import TSNE

port = os.environ.get('PORT', 3008)



### Lists, Constants, and Functions

NEW_CATEGORIES_SHORT = ['chair','table','dresser-chest','sofa','desk','bench','stool-ottoman','bar_cart',
                        'tv_stand-console','sideboard','bookshelves-bookcases-openshelves','cabinet',
                        'chaise','wardrobe','side_table-nightstand','vanities-washstands','bed',
                        'headboard','racks_hooks-ladder','crib']

NEW_CATEGORIES_LONG = ['chair','dining-chair','accent-chair','outdoor-chair','lounge-chair','rocking-chair',
                       'table','console-table','coffee-table','dining-table','outdoor-table','changing-table','entry-table','cocktail-table',
                       'dresser-chest',
                       'sofa','sectional-sofa','outdoor-sofa',
                       'desk','standing-desk','roll_top-desk','executive-desk',
                       'bench','bedroom-bench','entryway-bench','kitchen-bench','outdoor-bench',
                       'bar_cart',
                       'tv_stand-console',
                       'sideboard',
                       'stool-ottoman','bar-stool','accent-stool','outdoor-stool','outdoor-ottoman',
                       'bookshelves-bookcases-openshelves',
                       'cabinet','filing-cabinet','accent-cabinet','bar-cabinet',
                       'chaise','outdoor-chaise',
                       'wardrobe',
                       'side_table-nightstand',
                       'vanities-washstands','makeup-vanities',
                       'bed',
                       'headboard',
                       'racks_hooks-ladder',
                       'crib']

SHORT_RELEVANT_MATERIALS = ['metal','wood','plywood','stone','fabric','leathers',
                            'finished', 'unfinished', 'sanded', 'glass']

LONG_RELEVANT_MATERIALS = ['metal','brass','bronze','steel','aluminum','iron',
                           'wood','oak','hardwood','pine','birch','bamboo','walnut',
                           'mahogany','plywood','ash','patina','cedar','poplar','fir',
                           'stone','concrete','marble','granite',
                           'finished','unfinished','sanded','fabric','leathers','glass']

WOODS = ['oak','hardwood','pine','birch','bamboo','maple','cherry','walnut',
         'mahogany','plywood','ash','patina','cedar','poplar','fir','chesnut','sapele',
         'hickory','cypress','redwood','sycamore','rosewood']
METALS = ['brass','bronze','steel','aluminum','iron']
STONES = ['concrete','marble','quartz','granite']


MODEL_FEATURES = ['volume','max_dim']+NEW_CATEGORIES_LONG+LONG_RELEVANT_MATERIALS



def complete_cat_mat_labels(category, material):
    new_categories = []
    for c in category:
        new_categories.append(c)
        if c in NEW_CATEGORIES_LONG and c not in NEW_CATEGORIES_SHORT:
            common_cat = c.split('-')[1]
            to_append = [s for s in NEW_CATEGORIES_SHORT if common_cat in s][0]
            if to_append not in new_categories:
                new_categories.append(to_append)

    new_materials = []
    for m in material:
        new_materials.append(m)
        if m in LONG_RELEVANT_MATERIALS and m not in SHORT_RELEVANT_MATERIALS:
            if m in STONES and 'stone' not in new_materials:
                new_materials.append('stone')
            elif m in METALS and 'metal' not in new_materials:
                new_materials.append('metal')
            elif m in WOODS and 'wood' not in new_materials:
                new_materials.append('wood')

    return new_categories, new_materials


def hot_encode(volume, max_dim, category, material):
    test_case_features = []
    test_case_features.append(volume)
    test_case_features.append(max_dim)
    for r in MODEL_FEATURES[2:]:
        if r in category:
            test_case_features.append(1)
        elif r in material:
            test_case_features.append(1)
        else:
            test_case_features.append(0)

    print('hot encoded for regression model')
    return test_case_features


def embed_hot_encode(volume, max_dim, category, material):
    test_case_features = []
    for r in EMBED_FEATURE_LABEL:
        if r in category:
            test_case_features.append(1)
        elif r in material:
            test_case_features.append(1)
        elif r[0].isdigit() == True:
            if float(r.split('-')[0]) <= max_dim and float(r.split('-')[1]) > max_dim:
                test_case_features.append(1)
            elif float(r.split('-')[0]) <= volume and float(r.split('-')[1]) > volume:
                test_case_features.append(1)
            else:
                test_case_features.append(0)
        else:
            test_case_features.append(0)

    print('hot encoded for regression model')
    return test_case_features


def test_case(array):
    SAMPLE_INDX = len(prod_dataset_list)
    prod_dataset_list.append(array)

    test_X = []
    for a,b in zip(FEATURE_INDX, array):
        if b == 1:
            test_X.append([a, SAMPLE_INDX])

    embedding_test_case = []
    for x in test_X:
        false_data = 0

        ## True data
        embedding_test_case.append([x[0], x[1], 1])

        ## False data
#        if x[0] < 20: ## Category
#            FALSE = 8
#        elif x[0] >= 20 and x[0] < 30: ## Material
#            FALSE = 6
#        elif x[0] >= 30 and x[0] < 50: ## Max dim
#            FALSE = 3
#        elif x[0] >= 50 and x[0] < 70: ## Volume
#            FALSE = 6
    FALSE = 26

    while false_data < FALSE:
            rand1 = random.choice(embed_training_data)[0]
            rand2 = x[1]
            #rand1 = x[0]
            #rand2 = random.choice(embed_training_data)[1]
            if [rand1, rand2, 1] not in embed_training_data and [rand1, rand2, 1] not in embedding_test_case:
                embedding_test_case.append([rand1, rand2, -1])
                false_data+=1

    check2 = any(e[1] >= SAMPLE_INDX for e in embed_training_data)
    if check2 == False:
        for e in embedding_test_case:
            embed_training_data.append(e)


    ## Adding to max_dim_cat_list
    if len(prod_dataset_list) > len(max_dim_cat_list):
        for x,n in enumerate(array[33:53]):
            if n == 1:
                max_dim_cat_list.append(x)

    ## Adding to volume_cat_list
    if len(prod_dataset_list) > len(volume_cat_list):
        for x,n in enumerate(array[53:]):
            if n == 1:
                volume_cat_list.append(x)

    print("test case added to training data")

    
def find_similar(indx, weights, category):
    SAMPLE_WEIGHT = weights[indx]
    FIND_SIM_COLUMNS = ['items','cos_sim','prices','img_urls']+EMBED_FEATURE_LABEL
    prod_dataset = pd.DataFrame(ALL_EMBED_DATA['data_set'][:-1], columns=ALL_EMBED_DATA['data_set'][-1])
    cos_sims = []
    for x,w in enumerate(weights):
        cos_sim = np.dot(SAMPLE_WEIGHT,w)
        for l in LABELED_X:
            if l[1] == x:
                cat = l[0]
                break
        cos_sims.append([cos_sim, x, cat])

    #TOP_SIMS = sorted(cos_sims)[::-1][:6]
    if len(category) == 1:
        TOP_SIMS = [c for c in sorted(cos_sims)[::-1] if c[2] == category[0]][:6]
    elif len(category) == 2:        
        for cat in category:
            if cat in NEW_CATEGORIES_SHORT:
                TOP_SIMS = [c for c in sorted(cos_sims)[::-1] if c[2] == cat][:6]
                
    
    try:
        final = prod_dataset.iloc[[a[1] for a in TOP_SIMS]]
        final['cos_sim'] = [b[0] for b in TOP_SIMS]
        finish = final[FIND_SIM_COLUMNS]

    except:
        final = prod_dataset.iloc[[a[1] for a in TOP_SIMS[1:]]]
        final['cos_sim'] = [b[0] for b in TOP_SIMS[1:]]
        finish = final[FIND_SIM_COLUMNS]
        test_case_row = pd.DataFrame([['TEST_CASE', 1.000000, 'X', 'X']+prod_dataset_list[indx][:len(FIND_SIM_COLUMNS)-4]],
                                     columns=FIND_SIM_COLUMNS,
                                     index=[indx])

        finish = pd.concat([test_case_row, finish])

    return finish








### Code for API

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    INPUT = request.json
    
    ## Volume and Max Dim
    DIMENSIONS = INPUT['dimensions']
    MAX_DIM = 0
    VOLUME = 1
    for d in DIMENSIONS:
        try:
            VOLUME = VOLUME * float(d)
            if float(d) > MAX_DIM:
                MAX_DIM = float(d)
        except:
            return {'ERROR': 'Dimensions not acceptable'}

    ## Category
    CATEGORY = INPUT['category']
    for c in CATEGORY:
        if c not in MODEL_FEATURES:
            return {'ERROR': 'Category "' + str(c) + '" not in feature list'}

    ## Materials
    MATERIALS = INPUT['materials']
    for m in MATERIALS:
        if m not in MODEL_FEATURES:
            return {'ERROR': 'Material "' + str(m) + '" not in feature list'}
            

    ## Quantity
    QUANTITY = INPUT['quantity']
    try:
        QUANTITY = float(QUANTITY)
    except:
        return {'ERROR': 'Quantity not acceptable'}


    ## Completing missing cat and mat labels
    NEW_CATEGORY, NEW_MATERIAL = complete_cat_mat_labels(CATEGORY, MATERIALS)
    print(NEW_CATEGORY, NEW_MATERIAL)

    ## Reformat into one-hot
    HOT_SAMPLE = hot_encode(VOLUME, MAX_DIM, NEW_CATEGORY, NEW_MATERIAL)
    EMBED_SAMPLE = embed_hot_encode(VOLUME, MAX_DIM, NEW_CATEGORY, NEW_MATERIAL)
    #print(len(HOT_SAMPLE), len(EMBED_SAMPLE))

    ## Predict on Reg models
    if QUANTITY < 10:
        reg_model = MODEL_1
        RANGE = '+/- 33.34%'
    else:
        reg_model = MODEL_2
        RANGE = '+/- 54.74%'

    prediction = reg_model.predict(np.asarray(HOT_SAMPLE).reshape(1, -1))
    PREDICTED_PRICE = '$' + str(math.exp(prediction[0]))



    ## Embedding model
    test_case(EMBED_SAMPLE)

    ## Shuffling training data
    random.shuffle(embed_training_data)
    O = []
    H = []
    M = []
    for a, b, label in embed_training_data:
        O.append(a)
        H.append(b)
        M.append(label)

    O = np.array(O)
    H = np.array(H)
    M = np.array(M)
    
    ## Embedding model architecture
    EMBED_DIM = 40

    feature = tf.keras.layers.Input(name = 'feature', shape = [1])
    product = tf.keras.layers.Input(name = 'product', shape = [1])

    feature_embedding_layer = tf.keras.layers.Embedding(name = 'feature_embedding_layer',
                                                        input_dim = len(EMBED_FEATURE_LABEL), 
                                                        output_dim = EMBED_DIM)(feature)

    product_embedding_layer = tf.keras.layers.Embedding(name = 'product_embedding_layer',
                                                        input_dim = len(prod_dataset_list), 
                                                        output_dim = EMBED_DIM)(product)

    dot_product = tf.keras.layers.Dot(name = 'dot_product', 
                                      normalize = True, 
                                      axes = 2)([feature_embedding_layer, product_embedding_layer])

    reshaped = tf.keras.layers.Reshape(target_shape = [1])(dot_product)

    out = tf.keras.layers.Dense(1)(reshaped)

    cbr_model = tf.keras.Model(inputs = [feature, product], outputs = [out])
    cbr_model.compile(optimizer = tf.keras.optimizers.Adam(),
                            loss = 'mse')


    ## Stop loss funcion
    class GoodTrain(Callback):
        def __init__(self, stop_loss=0.2):
            super(Callback, self).__init__()
            self.stop_loss = stop_loss
        
        def on_epoch_end(self, epoch, logs=None):
            EPOCH_LOSS = logs.get('loss')
            if EPOCH_LOSS < self.stop_loss:
                self.model.stop_training = True
                self.good_training = False
            elif EPOCH_LOSS > self.stop_loss:
                self.good_training = True

            global GOOD_TRAINING
            GOOD_TRAINING = self.good_training

            print(self.good_training)
            
    ## Embedding model training
    GOOD_TRAINING = 0
    EPOCHS = 100
    BATCH_SIZE = len(prod_dataset_list)
    cbr_model.fit([O,H], M, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
                  #callbacks=[GoodTrain()])

    #print(GOOD_TRAINING)
    
    ## Extract embeddings
    product_embedding_layer = cbr_model.get_layer('product_embedding_layer')
    weights = product_embedding_layer.get_weights()[0]

    ## Normalised to find cosine similarity
    WEIGHTS = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    
    pickle_out = open("weights.pickle", 'wb')
    pickle.dump(WEIGHTS, pickle_out)
    pickle_out.close()
    
    ## Dimension Reduction
    NEW_DIM = 2
    reduced_dims = TSNE(NEW_DIM, metric = 'cosine').fit_transform(WEIGHTS)

    pickle_out = open("reduced_dims.pickle", 'wb')
    pickle.dump(reduced_dims, pickle_out)
    pickle_out.close()
    
    ## Finding 5 most similar for sample
    similars = find_similar(len(prod_dataset_list)-1, WEIGHTS, NEW_CATEGORY)[['items','cos_sim','prices','img_urls']]



    ## Making final return
    ITEMS = list(similars['items'])[1:]
    COS_SIMS = list(similars['cos_sim'])[1:]
    PRICES = list(similars['prices'])[1:]
    IMG_URLS = list(similars['img_urls'])[1:]
    SIM_PRODS_LIST = []
    for x,i in enumerate(ITEMS):
        to_append = {
            'product_name': i,
            'image': IMG_URLS[x],
            'price': PRICES[x],
            'similarity_score': round(COS_SIMS[x], 2)
            }
        
        SIM_PRODS_LIST.append(to_append)

    final = {
        'predicted_price': PREDICTED_PRICE,
        'suggested_range': RANGE,
        'similar_products': SIM_PRODS_LIST,
        }



    ### MongoDB
    client = MongoClient('mongodb+srv://projects-api:uuivOLLOnph08RSw@estimates.kuyn1.mongodb.net/estimates?retryWrites=true&w=majority')

    estimate_api_db = client['estimate_api_database']
    
    request_coll = estimate_api_db['request_collection']
    return_coll = estimate_api_db['return_collection']

    request_post = {
        '_id': 0,
        'dimensions': DIMENSIONS,
        'category': CATEGORY,
        'materials': MATERIALS,
        'quantity': QUANTITY
    }

    return_post = {
        '_id': 0,
        'volume': VOLUME,
        'full_category': NEW_CATEGORY,
        'full_materials': NEW_MATERIAL,
        'predicted_price': PREDICTED_PRICE,
        'suggested_range': RANGE,
        'similar_products': SIM_PRODS_LIST
    }

    request_coll.insert_one(request_post)
    return_coll.insert_one(return_post)

    return final
    





if __name__ == "__main__":    
    MODEL_1 = pickle.load(open("model_1.pickle", 'rb'))
    MODEL_2 = pickle.load(open("model_2.pickle", 'rb'))
    ALL_EMBED_DATA = pickle.load(open("all_embed_data.pickle", 'rb'))
    FEATURE_INDX = ALL_EMBED_DATA['feature_indx']
    EMBED_FEATURE_LABEL = ALL_EMBED_DATA['feature_label']
    LABELED_X = ALL_EMBED_DATA['labeled_x']

    embed_training_data = ALL_EMBED_DATA['embed_training_data']
    prod_dataset_list = ALL_EMBED_DATA['training']
    volume_cat_list = ALL_EMBED_DATA['volume_cat_list']
    max_dim_cat_list = ALL_EMBED_DATA['max_dim_cat_list']

    
    app.run(host='0.0.0.0',port=port)
    #app.run(host='0.0.0.0',port=port,debug=True)










