# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:00:03 2020

@author: Hasan
"""


from flask import Flask, render_template, request
import pickle
import numpy as np
import random

filename1 = 'gre-admission-rfc-model.pkl'
model = pickle.load(open(filename1, 'rb'))
filename2 = 'scaler.pkl'
scaler = pickle.load(open(filename2, 'rb'))
app = Flask(__name__)

@app.route('/home')
def home():
	return render_template('index.html')
@app.route('/')
def home1():
	return render_template('index.html')
@app.route('/about')
def about():
	return render_template('about.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        gre_score = int(request.form['gre_score'])
        TOFEL = int(request.form['TOFEL'])
        University_ranking = int(request.form['University_ranking'])
        SOP = int(request.form['SOP'])
        LOR = int(request.form['LOR'])
        CGPA = float(request.form['CGPA'])
        Research = int(request.form['Research'])
        suggestion=None
        suggestion1=None
        suggestion2=None
        data = np.array([[gre_score, TOFEL, University_ranking, SOP, LOR, CGPA, Research]])
        x_test_scaled=scaler.transform(data)
        my_prediction = model.predict(x_test_scaled)
        if ((my_prediction>.8) & (University_ranking>=4)):
            suggestion=random.choices(["Harvard","MIT","Stanford"],k=1)
            suggestion1=random.choices(["University of california","california Instittute","columbia university"],k=1)
            suggestion2=random.choices(["Yale","Washington university","Princeton"],k=1)
        elif ((my_prediction>.8) & (University_ranking==3)):
            suggestion=random.choices(["John Hopkins","University of chicago","University of California"],k=1)
            suggestion1=random.choices(["University of California LA","University of Pensylvania","University of Michigan"],k=1)
            suggestion2=random.choices(["Duke university","Cornell University","University of Michigan Ann Arbor"],k=1)
        elif ((my_prediction>.8) & (University_ranking==2)):
            suggestion=random.choices(["Rockefeller university","University of Souther california","Emory University"],k=1)
            suggestion1=random.choices(["Vanderbilt University","Michigan State University"],k=1)
            suggestion2=random.choices(["Brown university","Rice University"],k=1)
        elif ((my_prediction>.8) & (University_ranking==1)):
            suggestion=random.choices(["Brown university","Rice University"],k=1)
            suggestion1=random.choices(["University of Florida","University of Rochester","Tuft University"],k=1)
            suggestion2=random.choices(["University of Miami","Florida State University","University of Cincinnati"],k=1)
        elif ((my_prediction<.8) & (University_ranking>=4)):
            suggestion=random.choices(["John Hopkins","University of chicago","University of California"],k=1)
            suggestion1=random.choices(["University of California LA","University of Pensylvania","University of Michigan"],k=1)
            suggestion2=random.choices(["Duke university","Cornell University","University of Michigan Ann Arbor"],k=1)
        elif ((my_prediction<.8) & (University_ranking==3)):
            suggestion=random.choices(["Rockefeller university","University of Souther california","Emory University"],k=1)
            suggestion1=random.choices(["Vanderbilt University","Michigan State University"],k=1)
            suggestion2=random.choices(["Brown university","Rice University"],k=1)
        elif ((my_prediction<.8) & (University_ranking==2)):
            suggestion=random.choices(["Brown university","Rice University"],k=1)
            suggestion1=random.choices(["University of Florida","University of Rochester","Tuft University"],k=1)
            suggestion2=random.choices(["University of Miami","Florida State University","University of Cincinnati"],k=1)
        elif ((my_prediction<.8) & (University_ranking==1)):
            suggestion=random.choices(["Brown university","Rice University"],k=1)
            suggestion1=random.choices(["University of Florida","University of Rochester","Tuft University"],k=1)
            suggestion2=random.choices(["University of Miami","Florida State University","University of Cincinnati"],k=1)
        else:
            suggestion="will send u the recommendation over mail"
        prediction=int(my_prediction*100)
        
        return render_template('result.html',suggestion=suggestion,suggestion1=suggestion1,suggestion2=suggestion2,prediction_high=int(prediction+7),prediction_low=int(prediction-7))

if __name__ == '__main__':
	app.run(debug=True)