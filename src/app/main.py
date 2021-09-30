from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os


colunas = ['tamanho', 'ano', 'garagem']
# importando o modelo criado a partir do que foi importado do colab #
modelo = pickle.load(open('../../models/modelo.sav', 'rb'))

app = Flask(__name__)
# utiliza o os.environ.get para buscar uma variavel de ambiente 
# que vai subistitur o nome e senha (hardcode) #
# Importante criar as variaveis de ambiente pelo terminal usando export #
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

# faz as alterações necessárias para que o app esteja seguro de acordo com a parametrização de variavel #
basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API"


@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    polaridade = tb.sentiment.polarity
    return f"Polaridade: {polaridade}"


@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    # variavel que recebe os dados enviados pelo payload #
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

# host= 0.0.0.0 para rodar em varios ambientes #
app.run(debug=True, host='0.0.0.0')