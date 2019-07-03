#!/usr/bin/python
from flask import Flask
app = Flask(__name__)

@app.route("/")
def main_flask():
    with open('out.txt', 'r') as file:
        return(file.read())

if __name__ == '__main__':
    app.run('0.0.0.0')
