from flask import Flask, render_template, request
import json


labels = []

def read_labels () :
    # 从标签文件中读取所有标签
    path = __file__[:__file__.rfind('\\')]
    labelfile = path + '/static/labelfile.json'
    with open (labelfile, 'r', encoding='utf-8') as f:
        labels = json.loads(f.read())

    # 初始化标签文件
    if len(labels) == 0 :
        labels = [0 for i in range(4521)]
    
    return labels

def save_labels (labels) :
    path = __file__[:__file__.rfind('\\')]
    labelfile = path + '/static/labelfile.json'
    with open(labelfile, "w", encoding="utf-8") as f :
        f.write(json.dumps(labels))

labels = read_labels()

app = Flask(__name__)

@app.route('/<page>', methods = ["GET", "POST"])
def index(page) :
    if request.method == 'POST' : 
        for k in request.form :
            labels[int(k) - 1] = int(request.form[k])
            print('%d, %d' % (int(k), labels[int(k) - 1]))
            # print('%s, %s' % (k, request.form[k]))
        save_labels(labels)

    return render_template("index.html", page = int(page), labels = labels)



if __name__ == "__main__" :
    app.run(port=8000, host="localhost", debug=True)
