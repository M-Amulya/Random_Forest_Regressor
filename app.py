from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='template')
model = pickle.load(open('final_model.pickle', 'rb'))


@app.route("/", methods=["GET"])
def Home_page():
    return render_template("index.html")


standard_to = StandardScaler()


@app.route("/predict", methods=["POST"])
def prediction_page():
    if request.method == "POST":

        campaign_id = float(request.form["campaign_id"])
        subject_len = float(request.form["subject_len"])
        body_len = float(request.form["body_len"])
        mean_paragraph_len = float(request.form["mean_paragraph_len"])
        day_of_week = float(request.form["day_of_week"])
        product = float(request.form["product"])
        mean_CTA_len = float(request.form["mean_CTA_len"])
        target_audience = float(request.form["target_audience"])

        prediction = model.predict([[campaign_id, subject_len, body_len, mean_paragraph_len, day_of_week, product, mean_CTA_len, target_audience]])
        return render_template('result.html', result=prediction)

    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)
