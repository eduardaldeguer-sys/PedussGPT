from flask import Flask, request, jsonify
import json

# Aquí importas o defines tu MiniNN
# from mini_gpt import MiniNN

app = Flask(__name__)

# Inicializamos vocabulario y red neuronal
vocab = ["hola","adios","como","estas"]
nn = MiniNN(vocab)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    mensaje = data.get("mensaje","")
    
    # Predicción
    respuesta = nn.predecir(mensaje)
    
    # Si no entiende, pide respuesta al usuario (opcional)
    if respuesta==mensaje or respuesta=="No sé aún...":
        respuesta = data.get("enseñar","No sé qué responder aún")
        if respuesta!="No sé qué responder aún":
            nn.entrenar(mensaje,respuesta)
    
    return jsonify({"respuesta": respuesta})

if __name__=="__main__":
    app.run(debug=True)
