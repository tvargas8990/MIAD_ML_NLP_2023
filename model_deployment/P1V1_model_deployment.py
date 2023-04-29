#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

# Creamos en la cual vamos a predecir el precio del vehiculo segun las caracteristicas ingresadas por el usuario
def predict_price(pYear, pMileage, pState, pMake, pModel):
    # Importamos el modelo creado para predecir el precio de los carros 
    rf = joblib.load(os.path.dirname(__file__) + '/CarPricing_rf.pkl') 

    # Ahora, debemos hacer las transformaciones que hicimos para las variables categoricas 
    # Recordemos que para este modelo solo debemos usar las dummies que un RF inicial nos indico eran las mas relevantes 
    important_features = ['Year','Mileage','Model_Silverado','Model_Super','Make_GMC','Make_Lexus','Model_F-1504WD','Model_Wrangler','Model_Tahoe4WD','Make_BMW','Model_Escalade','Model_Suburban4WD','Make_Mercedes-Benz','Model_Rover','Make_Land','Make_Kia','Make_Ram','Model_Suburban2WD','Model_Tundra','Model_Tahoe2WD','Model_TahoeLT','Model_Sierra','Model_FusionSE','Make_Volkswagen','Model_TerrainFWD','Model_FocusSE','Model_CorvetteCoupe','Make_Porsche','Make_Hyundai','Model_Grand','Model_CruzeSedan','Make_Ford']

    # Inicializamos los valores en 0 
    lista_ceros = [0] * len(important_features)
    # Crear un diccionario con los nombres de columnas y los valores cero correspondientes
    zeros_dict = dict(zip(important_features, lista_ceros))

    # Crear un DataFrame de una fila con nombres de columna y valores cero
    df = pd.DataFrame([zeros_dict])

    # Vamos reemplazando los valores segun los valores ingresados por el usuario
    df['Year'] = pYear
    df['Mileage'] = pMileage
    df = df.apply(lambda col: col.apply(lambda x: 1 if pMake in col.name else x))
    df = df.apply(lambda col: col.apply(lambda x: 1 if pModel in col.name else x))
    # El modelo no usa los estados, por lo cual nos los vamos a tener en cuenta 
    # En este punto ya tenemos un dataframe con los valores que necesita el modelo 
    # Hacemos la prediccion y devolvemos el precio
    price = rf.predict(df)

    return price[0]

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