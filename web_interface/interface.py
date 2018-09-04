from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answers')
def answers():
    question = request.args.get('question')

    suggestions_list = []

    if question:
        r = requests.get('http://suggestqueries.google.com/complete/search?output=toolbar&hl=ru&q={}&gl=in'
                         .format(question))

        soup = BeautifulSoup(r.content, 'lxml')

        suggestions_data = soup.find_all('suggestion')
        for suggestion in suggestions_data:
            suggestions_list.append(suggestion.attrs['data'])

    return render_template('answers.html', suggestions=suggestions_list)


if __name__ == '__main__':
    app.run(debug=True)
