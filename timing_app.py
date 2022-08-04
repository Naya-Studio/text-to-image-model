
### API FOR TEXT-TO-IMAGE MODEL ###

from flask import Flask, request
import os

port = os.environ.get('PORT', 3008)

print('Timing test pre')

app = Flask(__name__)

@app.route('/timing', methods=['POST'])
def timing_test():
    
    print('Timing test start')

    print('Timing test finish')
    
    return 'time'


if __name__ == "__main__":
    print('Timing test PRE2')
    #app.run(host='0.0.0.0',port=port)
    app.run(host='0.0.0.0',port=port,debug=True)

























