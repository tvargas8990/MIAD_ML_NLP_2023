#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

# Creamos en la cual vamos a predecir el precio del vehiculo segun las caracteristicas ingresadas por el usuario
def predict_price(pYear, pMileage, pState, pMake, pModel):
    # Importamos el modelo creado para predecir el precio de los carros 
    rf = joblib.load(os.path.dirname(__file__) + '/CarPricing_rf.pkl') 
    # Debemos traer los promedios que construimos para asignarelos al modelo 
    working_directory = os.getcwd()
    print(working_directory)
    pathMaker = working_directory + '/maker_response_freq.csv'
    pathModel = working_directory + '/model_response_freq.csv'
    pathState = working_directory + '/state_response_freq.csv'
    dataMaker = pd.read_csv(pathMaker)
    dataModel = pd.read_csv(pathModel)
    dataState = pd.read_csv(pathState)
    
    # Vamos reemplazando los valores segun los valores ingresados por el usuario
    #df['Year'] = pYear
    #df['Mileage'] = pMileage
    # Obtenemos el promedio del State
    respState = -1
    # El estado debe tener un espacio al inicio para que funcione con el csv
    if not pState.startswith(" "):
        pState = " " + pState
    for s in range(0, len(dataState)):
        if dataState.loc[s]['State'] == pState:
            respState = dataState.loc[s]['Price']
    
    # Obtenemos el promedio del Maker
    respMaker = -1
    for m in range(0, len(dataMaker)):
        if dataMaker.loc[m]['Make'] == pMake:
            respMaker = dataMaker.loc[m]['Price']

    # Obtenemos el promedio del Modelo
    respModel = -1
    for l in range(0, len(dataModel)):
        if dataModel.loc[l]['Model'] == pModel:
            respModel = dataModel.loc[l]['Price']

    data = {'Year':[pYear], 'Mileage':[pMileage], 'State':[respState], 'Make':[respMaker], 'Model':[respModel]}
    df = pd.DataFrame(data)

    y_pred = rf.predict(df)
    return y_pred[0]
    

if __name__ == "__main__":
    
    if len(sys.argv) < 6:
        print('Please provide the 5 parameters: Year, Mileage, State, Make, Model')
        
    else:

        year = sys.argv[1]
        mileage = sys.argv[2]
        state = sys.argv[3]
        make = sys.argv[4]
        model = sys.argv[5]

        p1 = predict_price(year, mileage, state, make, model)
        
        print(f'Year:{year}, Mileage:{mileage}km, State:{state}, Make:{make}, Model:{model}')
        print('Predicted Price: ', p1)