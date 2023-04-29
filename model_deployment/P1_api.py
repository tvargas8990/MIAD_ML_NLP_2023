#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from P1V1_model_deployment import predict_price
from flask_cors import CORS

# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Vehicle Price Prediction API',
    description='Vehicle Price Prediction API')

ns = api.namespace('predict', 
     description='Vehicle Price Prediction')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Anio del vehiculo', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Distancia recorrida', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Estado', 
    location='args')

parser.add_argument(
    'Maker', 
    type=str, 
    required=True, 
    help='Productor del vehiculo', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Modelo del vehiculo', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        # Utiliza los argumentos adicionales en la función predict_proba()
        return {
         "result": predict_price(args['Year'], args['Mileage'], args['State'], args['Maker'], args['Model'])
        }, 200

# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
