# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:32:14 2021

@author: kusha
"""

# -*- coding: utf-8 -*-


from flask import Flask, render_template, request,flash,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager 
from sqlalchemy.sql.sqltypes import Date
import urllib.parse 
from datetime import datetime
from flask_login import UserMixin

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/cropwayydb'



#app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://cropwayy@cropwayydb:Messi#18@cropwayydb.mysql.database.azure.com/cropwayy"







app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SECRET_KEY'] = 'kush123'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

db.session.rollback()

login_manager = LoginManager()       
login_manager.login_view = 'login'
login_manager.init_app(app)


class Sign_up(UserMixin, db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    id = db.Column(db.Integer, primary_key=True)
    FullName = db.Column(db.String(80), nullable=False)
    UType = db.Column(db.String(20), nullable=False)
    Email = db.Column(db.String(30), nullable=False)
    Password = db.Column(db.String(20), nullable=False)
    Date = db.Column(db.String(20),nullable =  False)

class Farmer_info(UserMixin, db.Model):
   
    Email = db.Column(db.Integer, primary_key=True)
    Age = db.Column(db.Integer,nullable = False)
    Avatar = db.Column(db.LargeBinary, nullable=False) 
    ImageData = db.Column(db.Text, nullable=False)
    Phone = db.Column(db.Text, nullable=True)
    Description = db.Column(db.Text, nullable=True)
    Locality = db.Column(db.Text, nullable=True)
    Name = db.Column(db.String(128), nullable=False)
    Address = db.Column(db.Text, nullable=True)
    City = db.Column(db.Text, nullable=True)
    State = db.Column(db.Text, nullable=True)
    Pincode = db.Column(db.Integer, nullable=True)
    Experience = db.Column(db.Integer, nullable=True)
    Landsize = db.Column(db.Float, nullable=True)
    Crop = db.Column(db.Text, nullable=True)
    Status = db.Column(db.String, nullable=True)
    Tenure = db.Column(db.Integer, nullable=True)

class Corporate_info(UserMixin, db.Model):
   
    Email = db.Column(db.Integer, primary_key=True)
    Avatar = db.Column(db.LargeBinary, nullable=False) 
    ImageData = db.Column(db.Text, nullable=False)
    Phone = db.Column(db.Text, nullable=True)
    Description = db.Column(db.Text, nullable=True)
    Locality = db.Column(db.Text, nullable=True)
    Name = db.Column(db.String(128), nullable=False)
    Address = db.Column(db.Text, nullable=True)
    City = db.Column(db.Text, nullable=True)
    State = db.Column(db.Text, nullable=True)
    Reg = db.Column(db.String, nullable=True)
    YOE = db.Column(db.String,nullable=True)
    Pincode = db.Column(db.Integer, nullable=True)

class Wholesaler_info(UserMixin, db.Model):
   
    Email = db.Column(db.Integer, primary_key=True)
    Avatar = db.Column(db.LargeBinary, nullable=False) 
    ImageData = db.Column(db.Text, nullable=False)
    Phone = db.Column(db.Text, nullable=True)
    Description = db.Column(db.Text, nullable=True)
    Locality = db.Column(db.Text, nullable=True)
    Name = db.Column(db.String(128), nullable=False)
    Address = db.Column(db.Text, nullable=True)
    City = db.Column(db.Text, nullable=True)
    State = db.Column(db.Text, nullable=True)
    Pincode = db.Column(db.Integer, nullable=True)
 
class Retailer_info(UserMixin, db.Model):
   
    Email = db.Column(db.Integer, primary_key=True)
    Avatar = db.Column(db.LargeBinary, nullable=False) 
    ImageData = db.Column(db.Text, nullable=False)
    Phone = db.Column(db.Text, nullable=True)
    Description = db.Column(db.Text, nullable=True)
    Locality = db.Column(db.Text, nullable=True)
    Name = db.Column(db.String(128), nullable=False)
    Address = db.Column(db.Text, nullable=True)
    City = db.Column(db.Text, nullable=True)
    State = db.Column(db.Text, nullable=True)
    Pincode = db.Column(db.Integer, nullable=True)

class Post_info(UserMixin, db.Model):
   
    Email = db.Column(db.Integer,nullable=False)
    PostID = db.Column(db.Integer,primary_key=True) 
    UType = db.Column(db.String(20), nullable=False)
    Description = db.Column(db.Text, nullable=True)
    Price = db.Column(db.Float, nullable=True)
    Crop = db.Column(db.String(128), nullable=False)
    Date = db.Column(db.DateTime, nullable=False)
    Quantity = db.Column(db.Float, nullable=True)
    Status = db.Column(db.String(15), nullable=True)
    startDate = db.Column(db.String(25), nullable=False)
    endDate = db.Column(db.String(25), nullable=False)


class Crop_table(UserMixin, db.Model):
   
    cropid = db.Column(db.Integer,primary_key=True) 
    cropName = db.Column(db.String(20),nullable = False)
    MSP = db.Column(db.Integer,nullable = False) 

class Deals_info(UserMixin, db.Model):
   
    Invoice = db.Column(db.String(30),primary_key=True) 
    Crop = db.Column(db.String(20),nullable = False)
    bEmail = db.Column(db.String(30),nullable=False)
    sEmail = db.Column(db.String(30),nullable=False)
    Date = db.Column(db.String(30), nullable=True)
    Total = db.Column(db.Float, nullable=True)


class Contract_info(UserMixin, db.Model):
   
    PostID = db.Column(db.Integer,primary_key=True) 
    Crop = db.Column(db.String(20),nullable = False)
    cEmail = db.Column(db.String(30),nullable=False)
    fEmail = db.Column(db.String(30),nullable=False)
    #Date = db.Column(db.String(30), nullable=True)
    Status = db.Column(db.String(15), nullable=True)
    Total = db.Column(db.Float, nullable=True)
 
 






