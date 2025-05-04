from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # type: ignore
import mlflow.sklearn
import joblib
import os
from pymongo import MongoClient

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Optional test
try:
    models = genai.list_models()
    for m in models:
        print(f"{m.name}: {m.supported_generation_methods}")
    print("✅ API key is working. Models available:")
    print(models)
   

    
except Exception as e:
    print("❌ API call failed:", e)

# Initialize FastAPI
app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template rendering
templates = Jinja2Templates(directory="templates")

# Load model and transformers
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model_uri = "runs:/b0e618c550ca4208b216af807d7849e2/sentiment_model"
model = mlflow.sklearn.load_model(model_uri)

# History for display
history = []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/index", response_class=HTMLResponse)
async def sentiment_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_sentiment(request: Request, feedback: str = Form(...)):
    # Vectorize input
    review_vector = tfidf.transform([feedback])

    # Predict encoded label
    prediction_encoded = model.predict(review_vector)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # Predict probability
    prediction_proba = model.predict_proba(review_vector)

    # Get confidence of predicted class
    confidence = prediction_proba[0][prediction_encoded[0]] * 100  # As percentage

    # Set Bootstrap color
    color = "success" if prediction_label.lower() == "positive" else "danger" if prediction_label.lower() == "negative" else "warning"

    # Store feedback history
    history.insert(0, {
        "review": feedback,
        "result": prediction_label,
        "confidence": f"{confidence:.2f}%",  # Round nicely
        "color": color
    })
    if len(history) > 5:
        history.pop()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction_label,
        "confidence": f"{confidence:.2f}%",  # Pass to template
        "color": color,
        "history": history
    })


@app.get("/reviews", response_class=HTMLResponse)
async def get_reviews(request: Request):
    mongo_uri = os.getenv("MONGO_URI") 
    client = MongoClient(mongo_uri)
    db = client["Review-mlops"]
    collection = db["reviews"]

    # Fetch reviews, assuming each review has 'email' and 'review' keys
    cursor = collection.find({}, {"_id": 0})
    reviews = list(cursor)

    return templates.TemplateResponse("reviews.html", {
        "request": request,
        "reviews": reviews
    })


from fastapi.responses import JSONResponse

@app.post("/predict-review")
async def predict_review(review: str = Form(...), email: str = Form(...)):
    # Predict
    review_vector = tfidf.transform([review])
    prediction_encoded = model.predict(review_vector)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    prediction_proba = model.predict_proba(review_vector)
    confidence = prediction_proba[0][prediction_encoded[0]] * 100
    confidence_str = f"{confidence:.2f}%"
    sentiment_with_conf = f"{prediction_label} ({confidence_str})"

    # MongoDB connection
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["Review-mlops"]
    collection = db["reviews"]

    # Update by both email and review
    result = collection.update_one(
        {"email": email, "review": review},
        {"$set": {"sentiment": sentiment_with_conf}}
    )

    if result.matched_count == 0:
        # Insert new if not found
        collection.insert_one({
            "email": email,
            "review": review,
            "sentiment": sentiment_with_conf
        })
        
        # print(prediction_label)
        # print(confidence_str)


    return JSONResponse({
        "sentiment": prediction_label,
        "confidence": confidence_str
    })

import smtplib
from email.message import EmailMessage

@app.post("/send-reply")
async def send_reply(email: str = Form(...), message: str = Form(...)):
    try:
        sender_email = "vmmbaskaran@gmail.com"
        sender_password = os.getenv("SENDER_PASS") # use App Password

        msg = EmailMessage()
        msg.set_content(message)
        msg["Subject"] = "Response to Your Feedback"
        msg["From"] = sender_email
        msg["To"] = email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return JSONResponse({"success": True, "message": "Reply sent successfully."})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})
    
from google.generativeai import GenerativeModel, configure
from pymongo import MongoClient

import os

# Set Gemini API Key (make sure it's in your environment variables or .env file)
@app.post("/generate-reply")
async def generate_and_send_reply(email: str = Form(...)):
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["Review-mlops"]
    collection = db["reviews"]

    # Fetch review and sentiment
    review_data = collection.find_one({"email": email}, {"_id": 0, "review": 1, "sentiment": 1})
    if not review_data:
        return {"success": False, "error": "Email not found in database."}

    review = review_data["review"]
    sentiment = review_data["sentiment"]

    # Generate content using Gemini
    prompt = f"""The customer gave the following review: "{review}" which was classified as "{sentiment}".
Write a professional, polite email reply to the customer that reflects
this sentiment and addresses their concern or appreciation accordingly."""

    try:
        model = GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        reply = response.text

        # Send email using your existing send_reply logic
        sender_email = "vmmbaskaran@gmail.com"
        sender_password = os.getenv("SENDER_PASS")  # App Password for Gmail

        msg = EmailMessage()
        msg.set_content(reply)
        msg["Subject"] = "Response to Your Feedback"
        msg["From"] = sender_email
        msg["To"] = email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return {"success": True, "reply": reply}

    except Exception as e:
        return {"success": False, "error": str(e)}

    