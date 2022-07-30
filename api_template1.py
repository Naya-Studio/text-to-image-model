from flask import Flask, request
import json


################### Matching Algorithm #################################
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from array import *
import os
import re
import math
import scipy
from scipy import spatial

##########################################
import pandas as pd
import numpy as np
#########################################
client = MongoClient(os.environ.get('OLD_DB_ENDPOINT'))
naya_db = client['naya-app-database-v1']
R = 6371000 ##radius of earth in meters
port = os.environ.get('PORT', 3008)

## Geolocation API
import geopy
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="Your_Name")

############ DESIGNER FABRICATOR DATAFRAME ############
designerMakerFilter = {"user_type": {"$in": ["DESIGNER", "DESIGNER_MAKER", "MAKER"]}}
df = pd.DataFrame(list(naya_db["usermodels"].find(designerMakerFilter)))

keep_columns = ['_id', 'createdAt','description', 'detailed_location', 'email',
                'last_name', 'location', 'name', 'questions',  'updatedAt', 'user_type']

df = df[keep_columns]
df.questions.fillna(False, inplace=True)





############ DATA CLEANING ############

## Extracting info from MCQs
def extractQfromQs(_id):
    def extractId(row):
        if row and len(row) > 0:
            try:
                return next(q for q in row if q['id'] == _id)
            except:
                return False
        else: 
            return False
    return extractId

df['software'] = df.questions.apply(extractQfromQs("mcq11"))
df['dimensions'] = df.questions.apply(extractQfromQs("dimensions"))
df['work_experience'] = df.questions.apply(extractQfromQs("work-experience"))
df['materials_designer'] = df.questions.apply(extractQfromQs("mcq10"))
df['materials_maker'] = df.questions.apply(extractQfromQs("mcq1"))
df['capacity'] = df.questions.apply(extractQfromQs("mcq14"))
df['price_range'] = df.questions.apply(extractQfromQs("price-range"))

def make_profile_link(row):
    link_tail = str(row)
    link_head = 'https://ecosystem.naya.studio/user/'
    full_link = link_head+link_tail
    return full_link
    
df['profile_link'] = df._id.apply(make_profile_link)


## Cleaning dimension values
def returnFloats(d):
    returnValue = []
    for i in range(len(d)):
        try:
            returnValue.append(float(d[i]))
        except:
            returnValue.append(4)
    return sorted(returnValue)

df['dimensions'] = df.dimensions.apply(lambda row: row['dimensions'] if row else row)
df['dimensions']  = df.dimensions.apply(lambda row: [4,4,4] if not row else row)
df['dimensions']  = df.dimensions.apply(returnFloats)


## New adding location of each designer/maker else 'Boston, Ma'
df.detailed_location.fillna(False, inplace=True)
no_location = [d for d in df.detailed_location if d and d['city'] == 'Boston'][0]
df.detailed_location = df.detailed_location.apply(lambda row: row if row else no_location)
df['location_name'] = df.detailed_location.apply(lambda row: str(row['country']) if row['city'] == '' or row['city'] == None else str(row['city'])+', '+str(row['country']))


## Filling in any missing info
df["num_software"] = df.software.apply(lambda x: len(x["selected_options"]) if x else 0) ##number of useable software else 0
df.work_experience = df.work_experience.apply(lambda row: float(row['work_experience']) if row else 1) ##years of work exp else 1
df.capacity = df.capacity.apply(lambda row: row['selected_options'] if row and len(row['selected_options']) > 0 else ['1-10 pieces']) ##Maker's capacity else '1-10 pieces'
df['price_range'] = df['price_range'].apply(lambda row : row['price_range'] if row else [0, 1000]) ##Price range else $0-1000

lo_materials = []
for a,b in zip(list(df.materials_designer),list(df.materials_maker)):
    if a ==  False and b ==  False:
        mat = ['Wood']
    elif a == False:
        mat = b['selected_options']
    elif b == False:
        mat = a['selected_options']
    elif a and b:
        mat = a['selected_options']
    lo_materials.append(mat)
    
df['materials'] = lo_materials


## Extracting capacity values
def extractCapacity(row):
    return re.findall(r'\d+', row[0])

df.capacity = df.capacity.apply(extractCapacity)





############ FINAL ALGORITHM ############

def materialDistance(project_materials):
    def returnDistance(ecosystem_materials):
        distance = 0
        for material in project_materials:
            if material in ecosystem_materials:
                distance += 1
        distance = distance/len(project_materials) ##normalised the material distance output
        return distance
    return returnDistance


def dimensionDistance(project_dimensions):
    project_dimensions = sorted(project_dimensions, reverse=True)
    def returnDistance(ecosystem_dimensions):
        ecosystem_dimensions = sorted(ecosystem_dimensions, reverse=True)
        distance = 0
        for i, d in enumerate(project_dimensions):
            if d < ecosystem_dimensions[i]:
                distance += 1
        if distance == 3:
            final = 1
        else:
            final = 0
        return final
    return returnDistance


def capacityDistance(project_capacity):
    def returnDistance(ecosystem_capacity):
        distance = 0
        if len(ecosystem_capacity) == 2:
            if project_capacity < float(ecosystem_capacity[1]):
                distance += 1
                if project_capacity >= float(ecosystem_capacity[0]):
                    distance += 1
        else:
            try:
                if project_capacity < float(ecosystem_capacity[0]):
                    distance = 2
            except:
                distance = 0
        if distance == 2:
            final = 1
        else:
            final = 0
        return final
    return returnDistance


def priceDistance(project_budget):
    def returnDistance(ecosystem_range):
        distance = 0
        if project_budget < float(ecosystem_range[1]): ##either 1 or distance from number normalised
            distance += 1
            if project_budget >= float(ecosystem_range[0]):
                distance += 1
                
        if distance == 2:
            final = 1
        elif distance == 1:
            final = (float(ecosystem_range[0]) - project_budget)/float(ecosystem_range[0])
        else:
            final = (project_budget - float(ecosystem_range[1]))/project_budget
        return np.power(final, 2) ##returns distance**2
    return returnDistance


def locationDistance(location):
    def returnDistance(ecosystem_location):
        project_latitude = float(location['latitude'])
        project_longitude = float(location['longitude'])
    
        if ecosystem_location:
            ecosystem_latitude = float(ecosystem_location['lat'])
            ecosystem_longitude = float(ecosystem_location['lng'])
            
            lat1 = project_latitude*math.pi/180
            lat2 = ecosystem_latitude*math.pi/180
            dlat = (lat2-lat1)*math.pi/180
            dlon = (ecosystem_longitude-project_longitude)*math.pi/180
            
            a = math.sin(dlat/2)**2 + (math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            x = a**.5
            y = (1-a)**.5
            c = 2*math.atan2(x, y)
            d = R * c
            
            try:
##                return 1-(math.log10(d)/10)
                return d
            except:
##                return 1.0
                return 0.0
        
    return returnDistance


def norm_locationDistance(row):
    MAX = df.location_distance.max()
    normed_dist = 1-(row/MAX)
    
    return normed_dist




def Naya_MatchingAlgorithm(project_materials, project_units, project_budget, project_dimensions=False,
                           project_location=False, designer_route=True, chosen_weights=False):

    if chosen_weights == False:
        weights = {
            "materials": 1.0,
            "capacity": 1.0,
            "price": 1.0,
            "dimensions": 1.0,
            "location": 10
        }
        
    else:
        weights = chosen_weights

    if designer_route == True:
        features_returned = ['name', 'email', 'profile_link', 'price_range', 'capacity',
                             'materials', 'location_name', 'software', 'work_experience', 'feature_distances']
        
        df["materials_distance"] = df.materials.apply(materialDistance(project_materials))
        df["capacity_distance"] = df.capacity.apply(capacityDistance(project_units))
        df["price_distance"] = df.price_range.apply(priceDistance(project_budget))

        def designerDistance(row, selected_weights=['materials','capacity','price']):
            return sum([weights[s] * np.power(row[s+'_distance'], 2) for s in selected_weights])/sum([weights[s] for s in selected_weights])
        df["designer_distance"] = df.apply(designerDistance, axis=1)

        def allfeatureDistances(row, selected_weights=['materials','capacity','price']):
            all_feature_distances = {}
            for s in selected_weights:
                all_feature_distances[s+'_distance'] = row[s+'_distance']
            all_feature_distances['designer_distance'] = row['designer_distance']
            return all_feature_distances
        df["feature_distances"] = df.apply(allfeatureDistances, axis=1)
        
        df_designers = df[(df["user_type"] == "DESIGNER") | (df["user_type"] == "DESIGNER_MAKER")]   
        df_designers = df.sort_values('designer_distance', ascending=False).dropna()
        top_designers = df_designers.head(50) ## reduce to 5 on frontend
        
        designers = {}
        for x,indx in enumerate(top_designers.index):
            example = {}
            for f in features_returned:
                example[f] = list(top_designers[f])[x]
            designers[str(x+1)] = example

        return designers


    
    elif designer_route == False:
        features_returned = ['name', 'email', 'profile_link', 'price_range', 'capacity',
                             'materials', 'location_name', 'dimensions', 'work_experience', 'feature_distances']

        df["materials_distance"] = df.materials.apply(materialDistance(project_materials))
        df["capacity_distance"] = df.capacity.apply(capacityDistance(project_units))
        df["price_distance"] = df.price_range.apply(priceDistance(project_budget))
        df["dimensions_distance"] = df.dimensions.apply(dimensionDistance(project_dimensions))
        df["location_distance"] = df.detailed_location.apply(locationDistance(project_location))
        df["location_distance"] = df.location_distance.apply(norm_locationDistance)

        def makerDistance(row, selected_weights=['materials','capacity','price','dimensions','location']):
            return sum([weights[s] * np.power(row[s+'_distance'], 2) for s in selected_weights])/sum([weights[s] for s in selected_weights])
        df["maker_distance"] = df.apply(makerDistance, axis=1)

        def allfeatureDistances(row, selected_weights=['materials','capacity','price','dimensions','location']):
            all_feature_distances = {}
            for s in selected_weights:
                all_feature_distances[s+'_distance'] = row[s+'_distance']
            all_feature_distances['maker_distance'] = row['maker_distance']
            return all_feature_distances
        df["feature_distances"] = df.apply(allfeatureDistances, axis=1)
        
        df_makers = df[(df["user_type"] == "MAKER") | (df["user_type"] == "DESIGNER_MAKER")]
        df_makers = df.sort_values('maker_distance', ascending=False).dropna()
        top_makers = df_makers.head(50) ## reduce to 5 on frontend
        
        makers = {}
        for x,indx in enumerate(top_makers.index):
            example = {}
            for f in features_returned:
                example[f] = list(top_makers[f])[x]
            makers[str(x+1)] = example

        return makers
        
        





### CODE FOR API

app = Flask(__name__)

@app.route('/designer-match', methods=['POST'])
def designer_match():
    INPUT = request.json

    ## Project materials
    project_materials = INPUT['project_materials']

    ## Project Budget
    project_budget = INPUT['project_budget']

    ## Project_units
    project_units = INPUT['project_units']

##    ## Chosen weights
##    custom_weights = INPUT['custom_weights']

    ## Matchmaking algorithm
    designers = Naya_MatchingAlgorithm(project_materials, project_units, project_budget, designer_route=True)#, chosen_weights=custom_weights)

    return {'matched_designers': designers}



@app.route('/maker-match', methods=['POST'])
def maker_match():
    INPUT = request.json

    ## Project materials
    project_materials = INPUT['project_materials']

    ## Project dimensions
    input_dim = INPUT['project_dimensions']
    project_dimensions = [input_dim['length'], input_dim['width'], input_dim['height']]

    ## Project Budget
    project_budget = INPUT['project_budget']

    ## Project location
    project_location = INPUT['project_location']

    ## Project_units
    project_units = INPUT['project_units']

##    ## Chosen weights
##    custom_weights = INPUT['custom_weights']

    ## Matchmaking algorithm
    makers = Naya_MatchingAlgorithm(project_materials, project_units, project_budget, project_dimensions=project_dimensions,
                                    project_location=project_location, designer_route=False)#, chosen_weights=custom_weights)

    return {'matched_fabricators': makers}


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=port)
    #app.run(host='0.0.0.0',port=port,debug=True)
























