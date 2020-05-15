from flask import flash,Flask,render_template,request,session,logging,url_for,redirect
from passlib.hash import sha256_crypt
from sklearn.externals import joblib
import pandas as pd 
import numpy as np
import csv
from prettytable import PrettyTable
import pickle
import sklearn.tree
app = Flask(__name__,static_url_path='/static')
model = pickle.load(open('C:/Users/adars/Downloads/fyp/crop.pkl', 'rb'))
@app.route("/")
def home():
    session["log"]=False    
    return render_template("home.html")

@app.route("/login",methods=["GET","POST"])
def login():
    user="admin"
    pwd="admin123"
    if request.method == "POST":
        username = request.form.get("name")
        password=request.form.get("password")
        
        if (username!=user or password!=pwd):
            flash("You are not authenticated",category='error')
            return render_template("login.html")
        else:
            session["log"]=True
            flash("User authenticated",category="message")
            return render_template("second.html")


    return render_template("login.html")
#about
@app.route("/about")
def about():
    session["log"]=False
    return render_template("about.html")
#second
@app.route("/second")
def second():
    return render_template("second.html")
#logout
@app.route("/logout")
def logout():
    session.clear()
    flash("You are logged out","success")
    return redirect(url_for('login'))
#predict
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 10)
    
    session["log"]=True
    #return render_template('new.html',prediction_text ='Best Crop that can be grown is  {}'.format(output))
    
    if output == 1:
        return render_template("Wheat.html")
    elif output == 2:
        return render_template("Rice.html")
    elif output == 3:
        return render_template("Corn.html")
    elif output == 4:
        return render_template("Corn.html")
    elif output == 5:
        return render_template("Corn.html")
    elif output == 6:
        return render_template("Corn.html")
    elif output == 7:
        return render_template("Corn.html")
    elif output == 8:
        return render_template("Corn.html")
    elif output == 9:
        return render_template("Corn.html")
    elif output == 10:
        return render_template("Corn.html")   
    elif output == 11:
        return render_template("Corn.html") 
    elif output == 12:
        return render_template("Corn.html") 


if __name__ == "__main__":
    app.secret_key="finalyearproject"
    app.run(debug=True)
    
