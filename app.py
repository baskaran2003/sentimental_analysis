from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import mlflow.sklearn
import joblib
import os
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize FastAPI
app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template rendering
templates = Jinja2Templates(directory="templates")

# Load model and transformers
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model_uri = "runs:/68bafaa636684e5c83c3e561422224b2/sentiment_model"
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


from fastapi import Request, Form
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
    


'''
@app.post("/predict", response_class=HTMLResponse)
async def predict_sentiment(request: Request, feedback: str = Form(...)):
    # Step 1: Transform input
    review_vector = tfidf.transform([feedback])
    prediction_encoded = model.predict(review_vector)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    prediction_proba = model.predict_proba(review_vector)
    confidence = prediction_proba[0][prediction_encoded[0]] * 100

    review_array = review_vector.toarray()
    review_df = pd.DataFrame(review_array, columns=tfidf.get_feature_names_out())

    # Step 2: Define SHAP-friendly prediction wrapper
    def model_func(x):
        return model.predict_proba(x)

    # Step 3: Build SHAP explainer
    explainer = shap.Explainer(model_func, review_df)

    # Step 4: Get SHAP values
    shap_values = explainer(review_df)

    # Step 5: Save SHAP bar plot (first instance)
    plot_filename = f"static/shap_plot_{uuid.uuid4().hex[:8]}.png"
    plt.figure()
    shap.plots.bar(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    # Step 5: Get top contributing words
    word_contributions = sorted(
        zip(tfidf.get_feature_names_out(), shap_values.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_words = [word for word, _ in word_contributions[:3]]

    # Step 6: Setup styling
    color = "success" if prediction_label.lower() == "positive" else "danger" if prediction_label.lower() == "negative" else "warning"

    # Step 7: Add to history
    history.insert(0, {
        "review": feedback,
        "result": prediction_label,
        "confidence": f"{confidence:.2f}%",
        "color": color,
        "top_words": top_words
    })
    if len(history) > 5:
        history.pop()

    # Step 8: Return response
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction_label,
        "confidence": f"{confidence:.2f}%",
        "color": color,
        "top_words": top_words,
        "shap_plot": "/" + plot_filename,
        "history": history
    })
'''

    