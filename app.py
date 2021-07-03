# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:23:42 2021
@author: kusha
"""
from flask import Flask, render_template, request,flash,redirect,url_for,session
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from sqlalchemy.sql.expression import false
from auth import Sign_up,Farmer_info,Corporate_info,Wholesaler_info,Retailer_info,Post_info,Crop_table,Deals_info,Contract_info
from flask_session import Session
from functools import wraps
# Keras
import os
import tensorflow
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sqlalchemy.sql import text
import wikipedia
import joblib
from PIL import Image
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import numpy as np
from base64 import b64encode
import pytz
from sqlalchemy import and_
import datetime
from pytz import timezone
from sqlalchemy import DateTime,desc
import base64
from random import randint
from datetime import datetime,timezone
from io import BytesIO 
from flask_login import LoginManager 
from flask_login import login_user, current_user, logout_user,login_required
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask import session as login_session
from flask_login import UserMixin

app = Flask(__name__)



app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://cropwayy@cropwayydb:Messi#18@cropwayydb.mysql.database.azure.com/cropwayy"

#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/cropwayydb'

app.config['SECRET_KEY'] = 'kush123'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

db.create_all()

db.session.rollback()

bootstrap = Bootstrap(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.init_app(app)

#--------------------------------------------------------MODELS-----------------------------------------------#


wikipedia.set_lang("en")

UPLOAD_FOLDER = r'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

STATIC_DIR = os.path.abspath('static')

modelML = joblib.load(open("weights/crop_rec.pkl","rb"))
modelMLFert = joblib.load(open("weights/fertiliser.save","rb"))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


MODEL_PATH_PLANT = "weights/mdl_wts_own.hdf5"


# Load your trained model
modelPlant = load_model(MODEL_PATH_PLANT,compile = False)



#for new plant and paddy
def model_predict_plant(img_path, model):
    img = image.load_img(img_path, target_size=(150,150))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    preds = model.predict(x)

    return preds





#--------------------MODELS CODE START---------------------------#

def listToString(s): 
    
    str1 = " " 
    
    return (str1.join(s))

@app.route("/fert_rec",methods = ['GET','POST'])
@cross_origin()
def fert_rec():
   
    if request.method == 'POST':
        N = float(request.form["n"])

        P = float(request.form["p"])
        
        K = float(request.form["k"])

        temp = float(request.form["t"])
        hum = float(request.form["hum"])
        m = float(request.form["m"])
    
        soil = request.form['s']
        
        if(soil == 'a'):
            
            soil_red = 1
            soil_black =0
            soil_sandy =0
            soil_loomy = 0
            soil_clay = 0
            
        elif (soil == 'b'):
            
            soil_red = 0
            soil_black =1
            soil_sandy =0
            soil_loomy = 0
            soil_clay = 0
            
        elif (soil == 'c'):
            
            soil_red = 0
            soil_black =0
            soil_sandy =2
            soil_loomy = 0
            soil_clay = 0 
        elif (soil == 'd'):
            
            soil_red = 0
            soil_black =0
            soil_sandy =0
            soil_loomy = 3
            soil_clay = 0
        elif (soil == 'e'):
            
            soil_red = 0
            soil_black =0
            soil_sandy =0
            soil_loomy = 0
            soil_clay = 4
            
            
        crop = request.form['crop']
        if(crop == 'a'):
            cMaize =1
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'b'):
            cMaize =0
            cCotton=1
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'c'):
            cMaize =0
            cCotton=0
            cPaddy =1
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'd'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =1
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'e'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =1
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'f'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =1
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'g'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =1
            cWheat =0
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'h'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =1
            cMillets =0
            cOil=0
            cGround=0
            
        elif(crop == 'i'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =1
            cOil=0
            cGround=0
            
        elif(crop == 'j'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=1
            cGround=0
            
        elif(crop == 'k'):
            cMaize =0
            cCotton=0
            cPaddy =0
            cBarley =0
            cSugarcane =0
            cTobacco =0
            cPulses =0
            cWheat =0
            cMillets =0
            cOil=0
            cGround=1
        
     
        
        prediction = modelMLFert.predict([[temp,hum,m,N,P,K,soil_black,soil_clay,soil_loomy,soil_red,soil_sandy,cBarley,cCotton,cGround,cMaize ,
            cMillets ,
            cOil ,                              
            cPaddy ,
            cPulses ,
            cSugarcane ,
            cTobacco ,
            cWheat ]])
        
        
        prob_class = modelMLFert.predict_proba([[temp,hum,m,N,P,K,soil_black,soil_clay,soil_loomy,soil_red,soil_sandy,cBarley,cCotton,cGround,
            cMaize,
            cMillets ,
            cOil ,                              
            cPaddy ,
            cPulses ,
            cSugarcane ,
            cTobacco ,
            cWheat ]])
        
        pred_prob = np.max(prob_class)
 
        print(prediction)
        prediction = listToString(prediction)

        wikiInfo = wikipedia.summary('fertilizers')
        return render_template('result.html',pred_text_crop = prediction,infoDis = wikiInfo,prob = pred_prob) 
    return None

@app.route("/crop_rec",methods = ['GET','POST'])
@cross_origin()
def crop_rec():
   
    if request.method == 'POST':
        N = float(request.form["n"])

        P = float(request.form["p"])
        
        K = float(request.form["k"])

        temp = float(request.form["t"])
        hum = float(request.form["hum"])
        ph = float(request.form["ph"])
        rain = float(request.form["rain"])
        prediction = modelML.predict([[N,P,K,temp,hum,ph,rain]])
        
        prob_class = modelML.predict_proba([[N,P,K,temp,hum,ph,rain]])
        
        pred_prob = np.max(prob_class)
 
        print(prediction)
        prediction = listToString(prediction)

        try:
            wikiInfo = wikipedia.summary(prediction)
        except wikipedia.exceptions.PageError:  
            wikiInfo = "No information found"

            
        return render_template('result.html',pred_text_crop = prediction,infoDis = wikiInfo,prob = pred_prob) 
    return None
        
    

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/prediction', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)    
     
        
        
        preds = model_predict_plant(file_path, modelPlant)
        pred_prob = np.max(preds)
        pred_class = preds.argmax(axis=-1)            
        print(pred_class)
        print(pred_prob)
        result = pred_class[0]
        
        if(pred_prob < .80):
            res = "This image doesn't seem to be a crop"
            return render_template('result.html',pred_text_plant = res,prob = "N\L") 


        arr = ['Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn maize common rust',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy']

        for i in range(0,37):
            if (result == i):
                res = arr[i]

                
    else:
        return "Error"
    
    try:
        wikiInfo = wikipedia.summary(res)
    except wikipedia.exceptions.PageError:  
        wikiInfo = "No information found"            
                #return sol
    return render_template('result.html',pred_text_plant = res, infoDis = wikiInfo,prob = pred_prob)            



@app.route("/results")
def results():
    return render_template("result.html")




#--------------------MODELS CODE END---------------------------#




#------------------------------------CROPS------------------------------------------------#

allCropTuple =  Crop_table.query.with_entities(Crop_table.cropName).distinct().all()

allCrop = [item[0] for item in allCropTuple]



userType = "NULL"

def invoiceNumber():
    now_utc = datetime.now(timezone.utc)
    t = now_utc.microsecond
    rand = randint(1, 999)  
    return str(rand) + str(t)

@app.route('/', methods=['GET'])
def home():
    
    return render_template('index.html')

@app.route('/gallery', methods=['GET'])
def gallery():
    
    result = userType

    return render_template('gallery.html',flag = result)

@app.route('/aboutus', methods=['GET'])
def aboutus():
    
    return render_template('about.html',flag = userType)

@app.route('/contactus', methods=['GET'])
def contactus():
    
    return render_template('contactform.html',flag = userType)

@app.route('/login', methods=['GET'])
def login():
    
    return render_template('login.html',flag = userType)


@app.route('/signup', methods=['GET'])
def signup():
    
    return render_template('signup.html')


@app.route('/invoice/<int:id>', methods=['GET','POST'])
@login_required
def invoice(id):

    inv = invoiceNumber()

    
    IST = pytz.timezone('Asia/Kolkata')
    datetime_utc = datetime.now(IST)

    date = datetime_utc.strftime('%Y:%m:%d-%H:%M')

    u = text("select UType from Post_info where Post_info.PostID = :id")


    f = text("select farmer_info.Email, farmer_info.Phone,Farmer_info.Locality,farmer_info.Address,Farmer_info.State, Farmer_info.Pincode,Farmer_info.City from Cropwayydb.Post_info, Cropwayydb.Farmer_info where Post_info.Utype = 'farmer' AND post_info.PostID = :id and post_info.Email=farmer_info.Email")

    farmer = db.engine.execute(f, id = id).fetchall()

    print(farmer[0].Email)

    e = farmer[0].Email

    name = text("select Sign_up.FullName from cropwayydb.sign_up, cropwayydb.farmer_info where Sign_up.Email = :e")

    name = db.engine.execute(name, e = e).fetchall()

    w = text("select Sign_up.FullName, Wholesaler_info.Email, Wholesaler_info.Phone,Post_info.Crop,Post_info.Quantity,Wholesaler_info.Locality,Wholesaler_info.Address,Wholesaler_info.State, Wholesaler_info.Pincode,Wholesaler_info.City, Post_info.Price from Cropwayydb.Sign_up,Cropwayydb.Post_info, Cropwayydb.Wholesaler_info where Post_info.Utype = 'farmer' AND Sign_up.Email = Wholesaler_info.Email AND post_info.PostID = :id")

    info = db.engine.execute(w, id = id).fetchall()

   # print(info.Locality)

    total = int(info[0].Quantity) * int(info[0].Price) 
    
    return render_template('invoice.html',date = date,id = id,inv = inv,total = total,farmer = farmer[0],data = info[0],name = name[0])

@app.route('/rinvoice/<int:id>', methods=['GET','POST'])
@login_required
def rinvoice(id):

    IST = pytz.timezone('Asia/Kolkata')

    datetime_utc = datetime.now(IST)

    date = datetime_utc.strftime('%Y:%m:%d-%H:%M')

    u = text("select UType from Post_info where Post_info.PostID = :id")
    
    ut = db.engine.execute(u, id = id).fetchall()

    inv = invoiceNumber()
      

    if(ut[0].UType == "farmer"):
        
        f = text("select farmer_info.Email, farmer_info.Phone,Farmer_info.Locality,farmer_info.Address,Farmer_info.State, Farmer_info.Pincode,Farmer_info.City,Post_info.Crop,Post_info.Quantity,Post_info.Price from Cropwayydb.Post_info, Cropwayydb.Farmer_info where Post_info.Utype = 'farmer' AND post_info.PostID = :id and post_info.Email=farmer_info.Email")

        farmer = db.engine.execute(f, id = id).fetchall()
        
        e = farmer[0].Email

        name = text("select Sign_up.FullName from cropwayydb.sign_up, cropwayydb.farmer_info where Sign_up.Email = :e")

        name = db.engine.execute(name, e = e).fetchall()

    else:
        f = text("select wholesaler_info.Email, wholesaler_info.Phone,wholesaler_info.Locality,wholesaler_info.Address,wholesaler_info.State, wholesaler_info.Pincode,wholesaler_info.City,Post_info.Crop,Post_info.Quantity,Post_info.Price from Cropwayydb.Post_info, Cropwayydb.wholesaler_info where Post_info.Utype = 'wholesaler' AND post_info.PostID = :id and post_info.Email=wholesaler_info.Email")

        farmer = db.engine.execute(f, id = id).fetchall()

        e = farmer[0].Email

        name = text("select Sign_up.FullName from cropwayydb.sign_up, cropwayydb.wholesaler_info where Sign_up.Email = :e")

        name = db.engine.execute(name, e = e).fetchall()

    w = text("select Sign_up.FullName, Retailer_info.Email, Retailer_info.Phone, Retailer_info.Locality,Retailer_info.Address,Retailer_info.State, Retailer_info.Pincode,Retailer_info.City from Cropwayydb.Sign_up, Cropwayydb.Retailer_info where Sign_up.Email = Retailer_info.Email")

    info = db.engine.execute(w, id = id).fetchall()

   # print(info.Locality)

    total = int(farmer[0].Quantity) * int(farmer[0].Price) 
    
    return render_template('rinvoice.html',date = date,id = id,inv = inv,total = total,farmer = farmer[0],data = info[0],name = name[0])

@app.route('/flower', methods=['GET'])
@login_required
def flower():
    
     
    return render_template('flower.html')


@app.route('/wSellDeals', methods=['GET'])
@login_required
def wSellDeals():

    email = session['email']

    f = text("select * from Deals_info where sEmail = :email")

    user = db.engine.execute(f, email = email).fetchall()

    return render_template('wSellDeals.html',details = user)     



@app.route('/confirm_applications/<int:id>', methods=['GET','POST'])
@login_required
def confirm_applications(id):

    email = session['email']

    user = Contract_info.query.filter(and_(Contract_info.cEmail == email),(Contract_info.Status == 0),(Contract_info.PostID == id)).all()

   # f = text("select * from Contract_info where cEmail = :email AND Status = 0 AND PostID = :id")

    #user = db.engine.execute(f, email = email,id = id).fetchall()

    print(user[0].Status)

    user[0].Status = True

    #user.Status = True

    db.session.merge(user[0])
    db.session.flush()
    db.session.commit()

    flag = "You have confirmed the contract"

    return render_template('buyProduce.html',Flag = flag)   

@app.route('/fapplications', methods=['GET'])
@login_required
def fapplications():

    email = session['email']

    f = text("select * from Contract_info where fEmail = :email")

    user = db.engine.execute(f, email = email).fetchall()

    print(user[0].Status)

    if (user[0].Status) == False:
      
        status = "Pending"

    else:
      
        status = "Confirmed"

    print(status)

    return render_template('fapplications.html',details = user,status = status)   


@app.route('/applications', methods=['GET'])
@login_required
def applications():

    email = session['email']

    f = text("select * from Contract_info where cEmail = :email AND status = 0")

    user = db.engine.execute(f, email = email).fetchall()

    return render_template('applications.html',details = user)   

@app.route('/cdeals', methods=['GET','POST'])
@login_required
def cdeals():

    email = session['email']

    f = text("select * from Contract_info where cEmail = :email AND Status = 1")

    user = db.engine.execute(f, email = email).fetchall()

    print(user)

    return render_template('cdeals.html',details = user)         


@app.route('/fdeals', methods=['GET'])
@login_required
def fdeals():

    email = session['email']

    f = text("select * from Contract_info where fEmail = :email AND Status = 1")

    user = db.engine.execute(f, email = email).fetchall()

    print(user)

    return render_template('fdeals.html',details = user)     

@app.route('/rdeals', methods=['GET'])
@login_required
def rdeals():

    email = session['email']

    f = text("select * from Deals_info where bEmail = :email")

    user = db.engine.execute(f, email = email).fetchall()

    print(user)

    return render_template('rdeals.html',details = user) 

@app.route('/wdeals', methods=['GET'])
@login_required
def wdeals():

    email = session['email']

    f = text("select * from Deals_info where bEmail = :email")

    user = db.engine.execute(f, email = email).fetchall()

    print(user)

    return render_template('wdeals.html',details = user)  



@app.route('/post_farmer_contract/<int:id>', methods=['GET','POST'])
@login_required
def post_farmer_contract(id):

    fEmail = session['email']

    u = text("select * from Post_info where Post_info.PostID = :id")
    
    ut = db.engine.execute(u, id = id).fetchall()

    cEmail = ut[0].Email

    Crop = ut[0].Crop

    status = False

    IST = pytz.timezone('Asia/Kolkata')
    datetime_utc = datetime.now(IST)

    date = datetime_utc.strftime('%Y:%m:%d-%H:%M')

    Total = int(ut[0].Quantity) * int(ut[0].Price)

    #IST = timezone('UTC')     
 
    entry = Contract_info(Total = Total,Status = status, Crop = Crop, cEmail = cEmail, fEmail = fEmail,date = Date)
    db.session.add(entry)
    db.session.commit()

    msg = 'suc'

    return render_template('getContracts.html',crop_list = allCrop,msg = msg,isPost = True)

@app.route('/invoice_confirm/<int:id>/<date>/<number>', methods=['GET','POST'])
@login_required
def invoice_confirm(id,date,number):

    bEmail = session['email']

    user = text('select UType from Sign_up where Email = :bEmail')

    user = db.engine.execute(user, bEmail = bEmail).fetchall()


    u = text("select * from Post_info where Post_info.PostID = :id")
    
    ut = db.engine.execute(u, id = id).fetchall()

    sEmail = ut[0].Email

    Crop = ut[0].Crop

    Total = int(ut[0].Quantity) * int(ut[0].Price)     
 
    entry = Deals_info(Date = date,Invoice = number,Total = Total, Crop = Crop, sEmail = sEmail, bEmail = bEmail)
    db.session.add(entry)
    db.session.commit()


    if user[0].UType == 'wholesaler':
     
        return render_template('wdeals.html')

    elif user[0].UType == 'retailer':

        return render_template('rdeals.html')




@app.route('/tips', methods=['GET'])
@login_required
def tips():
    
    return render_template('tips.html')

@app.route('/wPost', methods=['GET'])
@login_required
def wPost():

    email = session["email"]

    if(Wholesaler_info.query.filter_by(Email=email).count() == 0):
    
        isInfo = False
    
    else:

        isInfo = True

    details = ""
    add = ""
    pin = ""
    state = "" 
    phone = ""

    details = Wholesaler_info.query.filter_by(Email=email).first()

    if details:

        add = details.Locality + " " + details.Address + " " + details.City

        pin = details.Pincode

        phone = details.Phone

        state = details.State

    return render_template('wPost.html',crop_list = allCrop,isinfo = isInfo,state = state,address = add, pincode = pin, phone = phone)


@app.route('/newPost', methods=['GET'])
@login_required
def newPost():

    email = session["email"]

    if(Farmer_info.query.filter_by(Email=email).count() == 0):
    
        isInfo = False
    
    else:

        isInfo = True

    details = ""
    add = ""
    pin = ""
    state = "" 
    phone = ""

    details = Farmer_info.query.filter_by(Email=email).first()

    if details:

        add = details.Locality + " " + details.Address + " " + details.City

        pin = details.Pincode

        phone = details.Phone

        state = details.State

    return render_template('newPost.html',crop_list = allCrop,isinfo = isInfo,state = state,address = add, pincode = pin, phone = phone)


@app.route('/newContract', methods=['GET'])
@login_required
def newContract():
    email = session["email"]

    details = Corporate_info.query.filter_by(Email=email).first()

    add = details.Locality + " " + details.Address + " " + details.City

    pin = details.Pincode

    phone = details.Phone

    state = details.State

    return render_template('newContract.html',crop_list = allCrop,state = state,address = add, pincode = pin, phone = phone)

    
@app.route('/sellProduce', methods=['GET'])
@login_required
def sellProduce():
    
    email = session["email"]


    if(Post_info.query.filter_by(Email=email).count() == 0):
    
        isPost = False
    
    else:

        isPost = True
    
    details = ""
    posts = ""
    add = ""
    name = ""

    details = Wholesaler_info.query.filter_by(Email=email).first()

    if details:

        name = Sign_up.query.filter_by(Email=email).first()

        add = details.Locality + " " + details.Address + " " + details.City

        posts = Post_info.query.filter(and_(Post_info.Email==email),(Post_info.Status==1)).order_by(desc(Post_info.PostID)).all()
    


    return render_template('sellProduce.html',add = add ,details = details,posts = posts,name = name,ispost = isPost)
    

@app.route('/buyProduce', methods=['GET'])
@login_required
def buyProduce():
    
    return render_template('buyProduce.html',crop_list = allCrop)

@app.route('/sellCrops', methods=['GET'])
@login_required
def sellCrops():
   
    email = session["email"]


    if(Post_info.query.filter_by(Email=email).count() == 0):
    
        isPost = False
    
    else:

        isPost = True
    
    details = ""
    posts = ""
    add = ""
    name = ""

    details = Farmer_info.query.filter_by(Email=email).first()

    if details:

        name = Sign_up.query.filter_by(Email=email).first()

        add = details.Locality + " " + details.Address + " " + details.City

        posts = Post_info.query.filter(and_(Post_info.Email==email),(Post_info.Status==1)).order_by(desc(Post_info.PostID)).all()
    


    return render_template('sellCrops.html',add = add ,details = details,posts = posts,name = name,ispost = isPost)

@app.route('/getContracts', methods=['GET'])
def getContracts():

    return render_template('getContracts.html',crop_list = allCrop)




@app.route('/contract_search_wholesaler', methods=['GET','POST'])
@login_required
def contract_search_wholesaler():

    email = session["email"]      

    state = request.form.get('state')

    crop  = request.form.get('crops')


    s = text("select Sign_up.FullName, Farmer_info.Email,Farmer_info.Name,Farmer_info.Phone,Post_info.Crop,Post_info.Description,Post_info.PostID,Post_info.Quantity,Farmer_info.Locality,Farmer_info.State,Farmer_info.Pincode,Farmer_info.City, Post_info.Price from Cropwayydb.Sign_up,Cropwayydb.Post_info, Cropwayydb.Farmer_info where Post_info.Utype = 'farmer' AND Sign_up.Email = Farmer_info.Email  AND Post_info.Email = Farmer_info.Email AND Post_info.Crop = :c AND Farmer_info.State = :s")

    Result = db.engine.execute(s,c = crop, s = state).fetchall()

    print(Result)

  
    if Result:
    
        isPost = True
    
    else:

        isPost = False


    return render_template('buyProduce.html',result = Result,ispost = isPost)

@app.route('/contract_search_retailer', methods=['GET','POST'])
@login_required
def contract_search_retailer():

    email = session["email"]      

    state = request.form.get('state')

    crop  = request.form.get('crops')

    view = request.form.get('view')
    print(view)

    if(view == 'farmer'):
        s = text("select Sign_up.FullName, Farmer_info.Email,Farmer_info.Name,Farmer_info.Phone,Post_info.Crop,Post_info.Description,Post_info.Description,Post_info.PostID,Post_info.Quantity,Farmer_info.Locality,Farmer_info.State,Farmer_info.Pincode,Farmer_info.City, Post_info.Price from Cropwayydb.Sign_up,Cropwayydb.Post_info, Cropwayydb.Farmer_info where Post_info.Utype = 'farmer' AND Sign_up.Email = Farmer_info.Email  AND Post_info.Email = Farmer_info.Email AND Post_info.Crop = :c AND Farmer_info.State = :s")

        Result = db.engine.execute(s,c = crop, s = state).fetchall()

        print(Result)
    else:
        s = text("select Sign_up.FullName, Wholesaler_info.Email,Wholesaler_info.Name,Wholesaler_info.Phone,Post_info.Crop,Post_info.Description,Post_info.PostID,Post_info.Quantity,Wholesaler_info.Locality,Wholesaler_info.State,Wholesaler_info.Pincode,Wholesaler_info.City, Post_info.Price from Cropwayydb.Sign_up,Cropwayydb.Post_info, Cropwayydb.Wholesaler_info where Post_info.Utype = 'wholesaler' AND Sign_up.Email = Wholesaler_info.Email  AND Post_info.Email = Wholesaler_info.Email AND Post_info.Crop = :c AND Wholesaler_info.State = :s")
        Result = db.engine.execute(s,c = crop, s = state).fetchall()

        print(Result)
  
    if Result:
    
        isPost = True
    
    else:

        isPost = False


    return render_template('getProduce.html',result = Result,ispost = isPost)

@app.route('/contract_search', methods=['GET','POST'])
@login_required
def contract_search():

    email = session["email"]      

    state = request.form.get('state')

    crop  = request.form.get('crops')


    s = text("select Sign_up.FullName, Corporate_info.Email,Corporate_info.Name,Corporate_info.Phone,Post_info.Date,Post_info.startDate,Post_info.endDate,Post_info.PostID,Post_info.Crop,Post_info.Description,Post_info.Quantity,Corporate_info.Locality,Corporate_info.State,Corporate_info.Pincode,Corporate_info.City, Post_info.Price from Cropwayydb.Sign_up,Cropwayydb.Post_info, Cropwayydb.Corporate_info where Post_info.Utype = 'corporate' AND Sign_up.Email = Corporate_info.Email  AND Post_info.Email = Corporate_info.Email AND Post_info.Crop = :c AND Corporate_info.State = :s")

    Result = db.engine.execute(s,c = crop, s = state).fetchall()

    print(Result)

  
    if Result:
    
        isPost = True
    
    else:

        isPost = False


    return render_template('getContracts.html',result = Result,ispost = isPost)



@app.route('/getProduce', methods=['GET'])
@login_required
def getProduce():
    
    return render_template('getProduce.html',crop_list = allCrop)


@app.route('/post_farmer_edit', methods=['GET','POST'])
@login_required
def post_farmer_edit():
    
    post_id = request.form['edit']

    desc = request.form.get('desc')

    price = request.form.get('price')

    quantity = request.form.get('qty')

    obj = Post_info.query.filter_by(PostID = post_id).first()

    print(desc)
    
    obj.Description = desc
    obj.Price = price
    obj.Quantity = quantity
    mspCheck = db.session.query(Crop_table.MSP).filter(Crop_table.cropName == obj.Crop).first()


    if(mspCheck.MSP >= int(price)):
        flash("*Price should be greater than or equal to MSP (Minimum Support Price)")
        return redirect('sellCrops')
        

    else:

        db.session.merge(obj)
        db.session.flush()
        db.session.commit()
        #db.session.commit()

        return render_template('sellCrops.html',crop_list = allCrop)

@app.route('/post_wholesaler_edit', methods=['GET','POST'])
@login_required
def post_wholesaler_edit():
    
    post_id = request.form['edit']

    desc = request.form.get('desc')

    price = request.form.get('price')

    quantity = request.form.get('qty')

    obj = Post_info.query.filter_by(PostID = post_id).first()

    print(desc)
    
    obj.Description = desc
    obj.Price = price
    obj.Quantity = quantity
    mspCheck = db.session.query(Crop_table.MSP).filter(Crop_table.cropName == obj.Crop).first()


    if(mspCheck.MSP >= int(price)):
        flash("*Price should be greater than or equal to MSP (Minimum Support Price)")
        return redirect('sellCrops')
        

    else:

        db.session.merge(obj)
        db.session.flush()
        db.session.commit()
        
        #db.session.commit()

        return render_template('sellProduce.html',crop_list = allCrop)

@app.route('/post_corporate_edit', methods=['GET','POST'])
@login_required
def post_corporate_edit():
    
    post_id = request.form['edit']

    desc = request.form.get('desc')

    price = request.form.get('price')

    quantity = request.form.get('qty')

    obj = Post_info.query.filter_by(PostID = post_id).first()

    print(desc)
    
    obj.Description = desc
    obj.Price = price
    obj.Quantity = quantity
    mspCheck = db.session.query(Crop_table.MSP).filter(Crop_table.cropName == obj.Crop).first()


    if(mspCheck.MSP >= int(price)):
        flash("*Price should be greater than or equal to MSP (Minimum Support Price)")
        return redirect('postContracts')
        

    else:

        print("editing")
        db.session.merge(obj)
        db.session.flush()
        db.session.commit()
        #db.session.commit()

        return render_template('postContracts.html',crop_list = allCrop)


@app.route('/post_farmer_delete/<int:id>', methods=['GET','POST','DELETE'])
@login_required
def post_farmer_delete(id):
    
    email = session["email"] 

    obj = db.session.query(Post_info).filter_by(PostID = id).first()

    if(Post_info.query.filter_by(Email=email).count() == 0):
    
        isPost = False
    
    else:

        isPost = True

    try:
        obj.Status = False
        db.session.merge(obj)
        db.session.commit()
        return render_template('sellCrops.html',ispost = isPost)

    except:
        return "There was a problem deleting the post"

@app.route('/post_corporate_delete/<int:id>', methods=['GET','POST','DELETE'])
@login_required
def post_corporate_delete(id):
    
    email = session["email"] 

    obj = db.session.query(Post_info).filter_by(PostID = id).first()

    if(Post_info.query.filter_by(Email=email).count() == 0):
    
        isPost = False
    
    else:

        isPost = True

    try:
        obj.Status = False
        db.session.merge(obj)
        db.session.commit()
        return render_template('postContracts.html',ispost = isPost)

    except:
        return "There was a problem deleting the post"

@app.route('/postContracts', methods=['GET'])
@login_required
def postContracts():
    
    email = session["email"]


    if(Post_info.query.filter_by(Email=email).count() == 0):
    
        isPost = False
    
    else:

        isPost = True
    
    details = ""
    posts = ""
    add = ""
    name = ""

    details = Corporate_info.query.filter_by(Email=email).first()

    if details:

        name = Sign_up.query.filter_by(Email=email).first()

        add = details.Locality + " " + details.Address + " " + details.City

        posts = Post_info.query.filter(and_(Post_info.Email==email),(Post_info.Status==1)).order_by(desc(Post_info.PostID)).all()
    


    return render_template('postContracts.html',add = add ,details = details,posts = posts,name = name,ispost = isPost)
    


@app.route('/fertilizer', methods=['GET'])
@login_required
def fertilizer():
    
    return render_template('fertilizer.html')

@app.route('/rDash', methods=['GET'])
@login_required
def rDash():
    
    return render_template('rdash.html')

@app.route('/govtScheme', methods=['GET'])
def govtScheme():
    
    return render_template('govtScheme.html')

@app.route('/wDash', methods=['GET'])
@login_required
def wDash():
    
    return render_template('wdash.html')

@app.route('/aDash', methods=['GET'])
@login_required
def aDash():
    
    return render_template('adash.html')

@app.route('/cropRec', methods=['GET'])
@login_required
def cropRec():
    
    return render_template('cropRec.html')    

@app.route("/fprofile", methods=['GET', 'POST'])
@login_required
def fprofile():

    email = session["email"]

    if(Farmer_info.query.filter_by(Email=email).count() == 0):
    
        isInfo = False
    
    else:

        isInfo = True

    detail = Farmer_info.query.get(email)

    image = ""

    #image = b64encode(detail.Avatar)

    if detail:

        if detail.Avatar:

            image = base64.b64encode(detail.Avatar).decode('ascii')
    

    return render_template('fprofile.html',crop_list = allCrop,isinfo = isInfo, details = detail,Image = image)

@app.route("/cprofile", methods=['GET', 'POST'])
@login_required
def cprofile():

    email = session["email"]

    detail = Corporate_info.query.get(email)

    image = ""

    #image = b64encode(detail.Avatar)

    if detail:

        if detail.Avatar:

            image = base64.b64encode(detail.Avatar).decode('ascii')

    return render_template('cprofile.html',details = detail)



@app.route("/rprofile", methods=['GET', 'POST'])
@login_required
def rprofile():

    email = session["email"]

    detail = Retailer_info.query.get(email)

    image = ""

    #image = b64encode(detail.Avatar)

    if detail:

        if detail.Avatar:

            image = base64.b64encode(detail.Avatar).decode('ascii')

    return render_template('rprofile.html',details = detail)

@app.route("/wprofile", methods=['GET', 'POST'])
@login_required
def wprofile():

    email = session["email"]

    if(Wholesaler_info.query.filter_by(Email=email).count() == 0):
    
        isInfo = False
    
    else:

        isInfo = True

    

    detail = Wholesaler_info.query.get(email)

    image = ""

    #image = b64encode(detail.Avatar)

    if detail:

        if detail.Avatar:

            image = base64.b64encode(detail.Avatar).decode('ascii')

    return render_template('wprofile.html', details = detail,isinfo = isInfo)


@app.route('/fDash', methods=['GET'])
@login_required
def fDash():
    return render_template('fdash.html',flag = userType)

@app.route('/farmBot', methods=['GET'])
def farmBot():
    
    return render_template('farmBot.html',flag = userType)

@app.route('/farmstore', methods=['GET'])
def farmstore():
    
    return render_template('farmstore.html',flag = userType)

@app.route('/cDash', methods=['GET'])
@login_required
def cDash():
    
    return render_template('cdash.html',flag = userType)

@app.route('/logout')
def logout():
    session.pop('email', None)
    
    logout_user()
    return redirect(url_for('login'))


@login_manager.user_loader
def load_user(id):
    return Sign_up.query.get(int(id))

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


#--------------------------------EDIT PROFILE START----------------------------#


def render_picture(data):

    render_pic = base64.b64encode(data).decode('ascii') 
    return render_pic

@app.route("/edit_farmer_submit", methods = ['GET', 'POST'])
def edit_farmer_submit():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)


    email = session["email"]

    age = request.form.get('age')

    phn = request.form.get('phn')

    desc = request.form.get('desc')

    city = request.form.get('city')

    pin = request.form.get('pin')

    state = request.form.get('state')

    exp = request.form.get('exp')

    crop = request.form.get('crops')

    status = request.form.get('status')

    add = request.form.get('add')

    tenure  = request.form.get('tenure')

    locality = request.form.get('loc')

    size = request.form.get('landsize')




    entry = Farmer_info(Age = age,City = city,Name=file.filename, Avatar = data, ImageData = render_file,Address = add,Locality = locality,Phone = phn, Description = desc,Email = email,Landsize = size,Crop = crop,Tenure = tenure, Status = status, Experience = exp, Pincode = pin, State = state)
    db.session.add(entry)
    db.session.commit()
    return redirect(url_for('fprofile'))



@app.route("/edit_wholesaler_submit", methods = ['GET', 'POST'])
def edit_wholesaler_submit():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)


    email = session["email"]

    phn = request.form.get('phn')

    desc = request.form.get('desc')

    city = request.form.get('city')

    pin = request.form.get('pin')

    state = request.form.get('state')

    add = request.form.get('add')

    locality = request.form.get('loc')

    entry = Wholesaler_info(City = city, Name=file.filename, Avatar = data, ImageData = render_file,Address = add,Locality = locality,Phone = phn, Description = desc,Email = email,Pincode = pin, State = state)
    db.session.add(entry)
    db.session.commit()
    return redirect(url_for('wprofile'))


@app.route("/post_farmer_submit", methods = ['GET', 'POST'])
def post_farmer_submit():

    email = session["email"]

    details = Sign_up.query.filter_by(Email=email).first()

    utype = details.UType

    crop = request.form.get('crops')

    print(crop)

    desc = request.form.get('desc')

    price = request.form.get('price')

    quantity = request.form.get('qty')

    IST = pytz.timezone('Asia/Kolkata')
    datetime_utc = datetime.now(IST)

    date = datetime_utc.strftime('%Y:%m:%d-%H:%M')

    mspCheck = db.session.query(Crop_table.MSP).filter(Crop_table.cropName == crop).first()

    print(mspCheck)

    if(mspCheck.MSP >= int(price)):
        flash("*Price should be greater than or equal to MSP (Minimum Support Price)")
        return redirect('newPost')
        

    else:

        entry = Post_info(Price = price,UType = utype ,Quantity = quantity,Crop = crop,Description = desc,Email = email, Date = date,Status = True)
        db.session.add(entry)
        db.session.commit()
        return redirect(url_for('sellCrops'))    

@app.route("/post_corporate_submit", methods = ['GET', 'POST'])
def post_corporate_submit():

    email = session["email"]

    details = Sign_up.query.filter_by(Email=email).first()

    utype = details.UType

    IST = pytz.timezone('Asia/Kolkata')

    datetime_utc = datetime.now(IST)

    date = datetime_utc.strftime('%Y:%m:%d-%H:%M')

    crop = request.form.get('crop')

    print(crop)

    desc = request.form.get('desc')

    price = request.form.get('price')

    startdate = request.form.get('start')

    enddate = request.form.get('end')

    print(enddate)

    quantity = request.form.get('qty')

    #select MSP from crop_table where cropname = paddy

    mspCheck = db.session.query(Crop_table.MSP).filter(Crop_table.cropName == crop).first()


    if(mspCheck.MSP >= int(price)):
        flash("*Price should be greater than or equal to MSP (Minimum Support Price)")
        return render_template('postContracts.html')
        

    else:

   # IST = timezone('UTC')


        entry = Post_info(Price = price,UType = utype ,startDate = startdate,endDate = enddate,Quantity = quantity,Crop = crop,Description = desc,Email = email, Date = date,Status = True)
        db.session.add(entry)
        db.session.commit()
        return redirect(url_for('postContracts'))

@app.route("/edit_retailer_submit", methods = ['GET', 'POST'])
def edit_retailer_submit():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)


    email = session["email"]

    phn = request.form.get('phn')

    desc = request.form.get('desc')

    city = request.form.get('city')

    pin = request.form.get('pin')

    state = request.form.get('state')

    add = request.form.get('add')

    locality = request.form.get('loc')

    entry = Retailer_info(City = city,Name=file.filename, Avatar = data, ImageData = render_file,Address = add,Locality = locality,Phone = phn, Description = desc,Email = email,Pincode = pin, State = state)
    db.session.add(entry)
    db.session.commit()
    return redirect(url_for('rprofile'))


@app.route("/edit_corp_submit", methods = ['GET', 'POST'])
def edit_corp_submit():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)

    email = session["email"]

    yoe = request.form.get('yoe')

    phn = request.form.get('phn')

    reg = request.form.get('reg')

    desc = request.form.get('desc')

    city = request.form.get('city')

    pin = request.form.get('pin')

    state = request.form.get('state')

    add = request.form.get('add')

    locality = request.form.get('loc')


    entry = Corporate_info(YOE = yoe,Reg = reg,City = city,Name=file.filename, Avatar = data, ImageData = render_file,Address = add,Locality = locality,Phone = phn, Description = desc,Email = email,Pincode = pin, State = state)
    db.session.add(entry)
    db.session.commit()
    return redirect(url_for('cprofile'))






@app.route("/edit_farmer_pass_submit", methods = ['GET', 'POST'])
def edit_farmer_pass_submit():


    email = session["email"]

    print(email)

    detail = Sign_up.query.filter_by(Email=email).first()

    print(detail)

    old = request.form.get('old')

    new = request.form.get('new')

    if check_password_hash(detail.Password, old):


        detail.Password = generate_password_hash(new, method='sha256')
        db.session.merge(detail)
        db.session.commit()
        

        return redirect(url_for('fprofile'))
    
    return "Old password does not match, Please try again"


#-----------------------------------EDIT PROFILE END-------------------------------#



#----------------------------------LOGIN START----------------------------#

@app.route("/login_submit", methods = ['GET', 'POST'])
def login_submit():

    email = request.form.get('email')
    session['email'] = email
    password = request.form.get('password')
    #remember = True if request.form.get('remember') else False

 
    try:
        user = Sign_up.query.filter_by(Email=email).first()
    except Exception :
        session.rollback()
    
    if not user:
        return "<h1> User does not exist, Please create an account and try again </h1>"



    global userType 

    userType = user.UType

    
      
    if user:
        if check_password_hash(user.Password, password):
            login_user(user)
            
            if (user.UType == 'farmer'):
            
                return redirect(url_for('fDash'))
            
            elif(user.UType == 'corporate'):
                return redirect(url_for('cDash'))
            elif(user.UType == 'retailer'):
                return redirect(url_for('rDash'))
            elif(user.UType == 'wholesaler'):
                return redirect(url_for('wDash'))  
            elif(user.UType == 'admin'):
                return redirect(url_for('aDash'))      
            else:
                return redirect(url_for('aDash'))
    

    return '<h1>Invalid username or password</h1>'
   


    return render_template('login.html')




#check_password_hash(user.password, password)
@app.route("/signup_submit", methods = ['GET', 'POST'])
def signup_submit():
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        password = request.form.get('password')
        cpassword = request.form.get('cpassword')
        email = request.form.get('email')
        user_type = request.form.get('type')
        
        if(len(password) < 6 or len(password) > 15):
            flash("Password length must be between 6 to 15 Characters")
            return redirect(url_for('signup'))
        
        if (password!=cpassword):
            flash("Password Doesn't Match")
            return redirect(url_for('signup'))

        IST = pytz.timezone('Asia/Kolkata')
        
        datetime_utc = datetime.now(IST)

        date = datetime_utc.strftime('%Y:%m:%d-%H:%M')
        
        
        user = Sign_up.query.filter_by(Email=email).first()

        if user:
            flash('Email address already exists.')
            return redirect(url_for('signup'))
        
        
        entry = Sign_up(FullName =name,UType = user_type, Date= date,Email = email,Password = generate_password_hash(password, method='sha256'))
        db.session.add(entry)
        db.session.commit()
        return redirect(url_for('login'))
    
    #----------------------------------LOGIN END----------------------------#
    
#Error 404     
@app.errorhandler(404) 
def invalid_route(e): 
    return render_template('404.html')

@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()

#Error 500
@app.errorhandler(500) 
def invalid_route(e): 
    return render_template('500.html')


if __name__ == '__main__':
   app.run()