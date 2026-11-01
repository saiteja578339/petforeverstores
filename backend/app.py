"""Pet Fur-Ever demo API.

Run with: uvicorn app:app --reload --port 8000

Scikit-learn models are trained from labeled/history data submitted to the
training endpoints. No fabricated training set is used. CLIP is an optional
dependency and loads lazily when /api/vision/clip-search is called.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("PET_DB_PATH", str(BASE_DIR / "data" / "listings.sqlite3")))
UPLOAD_DIR = Path(os.getenv("PET_UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Pet Fur-Ever Marketplace API", version="1.1.0")
app.mount("/media", StaticFiles(directory=UPLOAD_DIR), name="media")

allowed_origins = [
    item.strip()
    for item in os.getenv(
        "PET_CORS_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000,null",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

fake_listing_model: RandomForestClassifier | None = None
trust_model: RandomForestClassifier | None = None
clip_processor: Any = None
clip_model: Any = None


@contextmanager
def db_connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                breed TEXT NOT NULL,
                age TEXT NOT NULL,
                price TEXT NOT NULL,
                location TEXT NOT NULL,
                description TEXT NOT NULL,
                images_json TEXT NOT NULL DEFAULT '[]',
                trust_score INTEGER NOT NULL,
                risk_score INTEGER NOT NULL,
                requires_review INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        listing_columns = {row[1] for row in conn.execute("PRAGMA table_info(listings)")}
        for column, declaration in (
            ("sex", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("food", "TEXT NOT NULL DEFAULT ''"),
            ("breed_details", "TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in listing_columns:
                conn.execute(f"ALTER TABLE listings ADD COLUMN {column} {declaration}")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS interactions (
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                rating REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, item_id)
            )"""
        )


@app.on_event("startup")
def startup() -> None:
    init_db()


class ListingSignals(BaseModel):
    name: str = ""
    category: str = ""
    breed: str = ""
    age: str = ""
    price: str = ""
    description: str = ""
    photo_count: int = Field(default=0, ge=0, le=20)


class FakeTrainingExample(BaseModel):
    listing: ListingSignals
    is_fake: bool


class FakeTrainingRequest(BaseModel):
    examples: list[FakeTrainingExample] = Field(min_length=10, max_length=50000)


class TrustProfile(BaseModel):
    student_email_verified: bool = False
    identity_verified: bool = False
    phone_verified: bool = False
    completed_transactions: int = Field(default=0, ge=0, le=10000)
    response_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    reports: int = Field(default=0, ge=0, le=10000)
    listing_complete: bool = False
    health_documents: bool = False


class TrustTrainingExample(BaseModel):
    profile: TrustProfile
    is_trusted: bool


class TrustTrainingRequest(BaseModel):
    examples: list[TrustTrainingExample] = Field(min_length=10, max_length=50000)


class DemandPoint(BaseModel):
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    count: float = Field(ge=0)


class DemandForecastRequest(BaseModel):
    category: str
    history: list[DemandPoint] = Field(min_length=1, max_length=120)
    horizon: int = Field(default=3, ge=1, le=12)


class PetCandidate(BaseModel):
    id: str | int
    name: str
    category: str
    breed: str = ""
    age: str = ""
    price: str = ""
    images: list[str] = Field(default_factory=list)
    trust_score: int = Field(default=0, ge=0, le=100)


class Interaction(BaseModel):
    user_id: str
    item_id: str
    rating: float = Field(ge=0, le=5)


class InteractionWrite(Interaction):
    user_id: str = Field(min_length=1, max_length=100)
    item_id: str = Field(min_length=1, max_length=100)


class PreferenceProfile(BaseModel):
    space: str = "apartment"
    activity: str = "low"
    experience: str = "first"
    budget: str = "low"


class RecommendationRequest(BaseModel):
    user_id: str = "guest"
    preferences: PreferenceProfile = Field(default_factory=PreferenceProfile)
    interactions: list[Interaction] = Field(default_factory=list, max_length=100000)
    candidates: list[PetCandidate] = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class AdoptionReadinessRequest(BaseModel):
    home_type: Literal["apartment", "house", "renter"] = "apartment"
    daily_time: Literal["under_1", "1_2", "2_plus"] = "1_2"
    monthly_budget: Literal["under_3000", "3000_6000", "over_6000"] = "3000_6000"
    household_agrees: bool = False
    landlord_approval: bool = False
    emergency_fund: bool = False
    primary_caregiver: bool = False
    travel_care_plan: bool = False


class WelcomePlanRequest(BaseModel):
    pet_name: str = Field(min_length=1, max_length=100)
    category: Literal["puppy", "kitten", "rabbit", "bird"]
    age: str = Field(default="", max_length=40)


class ListingCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=40)
    breed: str = Field(min_length=1, max_length=100)
    age: str = Field(min_length=1, max_length=40)
    sex: Literal["male", "female", "unknown"] = "unknown"
    food: str = Field(default="", max_length=300)
    breed_details: str = Field(default="", max_length=1000)
    price: str = Field(min_length=1, max_length=40)
    location: str = Field(default="Local Listing", max_length=100)
    description: str = Field(min_length=1, max_length=4000)
    images: list[str] = Field(default_factory=list, max_length=10)
    trust_score: int = Field(default=0, ge=0, le=100)
    risk_score: int = Field(default=0, ge=0, le=100)
    requires_review: bool = False


def _price_value(price: str) -> float:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", price or "")
    return float(match.group(0).replace(",", "")) if match else 0.0


def fake_features(item: ListingSignals) -> list[float]:
    text = f"{item.name} {item.breed} {item.description}".lower()
    risky_phrases = ["wire money", "gift card", "crypto", "deposit first", "pay immediately", "urgent sale"]
    health_terms = ["vaccin", "vet", "health", "deworm", "medical"]
    suspicious_count = sum(text.count(phrase) for phrase in risky_phrases)
    has_health = any(term in text for term in health_terms)
    return [
        min(len(item.description), 4000),
        suspicious_count,
        min(_price_value(item.price), 1000000),
        item.photo_count,
        int(has_health),
        int(bool(item.breed.strip())),
        int(bool(item.age.strip())),
    ]


def fake_rule_score(item: ListingSignals) -> tuple[int, list[str]]:
    risk = 0
    signals: list[str] = []
    description = item.description.strip()
    text = f"{item.name} {item.breed} {description}".lower()
    price = _price_value(item.price)
    if len(description) < 45:
        risk += 20
        signals.append("Short description")
    if re.search(r"wire money|gift card|crypto|deposit first|pay immediately|urgent sale", text):
        risk += 35
        signals.append("High-pressure or unusual payment wording")
    if item.photo_count == 0:
        risk += 15
        signals.append("No seller photo attached")
    if price == 0:
        risk += 15
        signals.append("Price needs clarification")
    elif price < 500 or price > 200000:
        risk += 10
        signals.append("Price is outside the broad review range")
    if not re.search(r"vaccin|vet|health|deworm|medical", description, re.I):
        risk += 10
        signals.append("No health-care details mentioned")
    return min(risk, 100), signals


def trust_features(profile: TrustProfile) -> list[float]:
    return [
        int(profile.student_email_verified),
        int(profile.identity_verified),
        int(profile.phone_verified),
        min(profile.completed_transactions, 100),
        profile.response_rate,
        min(profile.reports, 100),
        int(profile.listing_complete),
        int(profile.health_documents),
    ]


def trust_rule_score(profile: TrustProfile) -> tuple[int, list[str]]:
    score = 35
    reasons: list[str] = []
    checks = [
        (profile.student_email_verified, 15, "Student email verified"),
        (profile.identity_verified, 15, "Identity verified"),
        (profile.phone_verified, 10, "Phone verified"),
        (profile.completed_transactions > 0, 10, "Completed transaction history"),
        (profile.response_rate >= 0.8, 5, "Strong response rate"),
        (profile.listing_complete, 5, "Complete listing details"),
        (profile.health_documents, 5, "Health documents supplied"),
    ]
    for passed, points, label in checks:
        if passed:
            score += points
            reasons.append(label)
    if profile.reports:
        score -= min(40, profile.reports * 10)
        reasons.append(f"{profile.reports} report(s) on record")
    if not reasons:
        reasons.append("No verification or history signals supplied")
    return max(0, min(100, score)), reasons


def parse_month(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid month '{value}'; expected YYYY-MM.") from exc


def month_after(value: str, offset: int) -> str:
    dt = parse_month(value)
    month_index = dt.year * 12 + (dt.month - 1) + offset
    year, month_zero = divmod(month_index, 12)
    return f"{year:04d}-{month_zero + 1:02d}"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models": {
            "fake_listing": "trained" if fake_listing_model else "rules_only; train with labeled data",
            "trust": "trained" if trust_model else "rules_only; train with labeled data",
            "clip": "loaded" if clip_model is not None else "optional; lazy load on first CLIP request",
        },
    }


@app.post("/api/vision/clip-search")
async def clip_search(
    image: UploadFile = File(...),
    candidates_json: str = Form(...),
    candidate_images: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Compare the query against actual listing photos with CLIP image embeddings."""
    global clip_processor, clip_model
    try:
        from PIL import Image, UnidentifiedImageError
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="CLIP is not installed. Run python -m pip install -r requirements-clip.txt, then restart the API.",
        ) from exc
    try:
        candidates = json.loads(candidates_json)
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 100:
            raise ValueError("Provide between 1 and 100 candidates.")
        if len(candidates) != len(candidate_images):
            raise ValueError("Each candidate needs exactly one listing photo.")
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 8 MB upload limit.")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        candidate_pictures: list[Image.Image] = []
        total_candidate_bytes = 0
        for candidate_file in candidate_images:
            photo_bytes = await candidate_file.read(2 * 1024 * 1024 + 1)
            total_candidate_bytes += len(photo_bytes)
            if len(photo_bytes) > 2 * 1024 * 1024 or total_candidate_bytes > 24 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Listing photos exceed the image-search upload limit.")
            candidate_pictures.append(Image.open(io.BytesIO(photo_bytes)).convert("RGB"))
    except HTTPException:
        raise
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image or candidates: {exc}") from exc
    if clip_model is None or clip_processor is None:
        model_name = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
        try:
            clip_processor = CLIPProcessor.from_pretrained(model_name)
            clip_model = CLIPModel.from_pretrained(model_name)
            clip_model.eval()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not load CLIP model '{model_name}': {exc}") from exc
    species = ["dog", "cat", "rabbit", "bird"]
    species_prompts = [f"A clear photograph of a {label}." for label in species]
    inputs = clip_processor(text=species_prompts, images=pil_image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        query_result = clip_model(**inputs)
        query_embedding = query_result.image_embeds
        species_probabilities = torch.softmax(query_result.logits_per_image[0], dim=0).cpu().numpy()
        candidate_inputs = clip_processor(images=candidate_pictures, return_tensors="pt")
        candidate_vision = clip_model.vision_model(pixel_values=candidate_inputs["pixel_values"])
        candidate_embeddings = clip_model.visual_projection(candidate_vision.pooler_output)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=-1)
        similarities = (candidate_embeddings @ query_embedding[0]).cpu().numpy()

    species_index = int(np.argmax(species_probabilities))
    detected_category = species[species_index]
    confidence = float(species_probabilities[species_index])
    category_map = {"puppy": "dog", "dog": "dog", "kitten": "cat", "cat": "cat", "rabbit": "rabbit", "bird": "bird"}
    best_similarity = float(np.max(similarities)) if len(similarities) else 0.0
    ranked = []
    for item, similarity in zip(candidates, similarities):
        pet_category = category_map.get(str(item.get("category", "")).lower())
        if confidence >= 0.45 and pet_category != detected_category:
            continue
        if float(similarity) < 0.18 or float(similarity) < best_similarity - 0.08:
            continue
        match_score = max(1, min(99, round((float(similarity) - 0.15) / 0.35 * 100)))
        ranked.append({"id": item.get("id"), "score": match_score, "similarity": round(float(similarity), 4)})
    ranked.sort(key=lambda row: row["similarity"], reverse=True)
    for item in ranked:
        item.pop("similarity", None)
    return {
        "algorithm": "CLIP image-to-image cosine similarity on actual listing photos",
        "model": os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
        "detected_category": detected_category if confidence >= 0.45 else None,
        "category_confidence": round(confidence, 3),
        "results": ranked[:3],
    }


@app.post("/api/fake-listings/score")
def score_fake_listing(item: ListingSignals) -> dict[str, Any]:
    rule_score, signals = fake_rule_score(item)
    ml_probability = None
    if fake_listing_model is not None:
        ml_probability = float(fake_listing_model.predict_proba([fake_features(item)])[0][1])
        score = round(0.5 * rule_score + 0.5 * 100 * ml_probability)
        method = "trained Scikit-learn classifier + rule signals"
    else:
        score = rule_score
        method = "rule signals; train the Scikit-learn classifier with labeled examples"
    return {
        "risk_score": score,
        "requires_review": score >= 50,
        "signals": signals,
        "model_fake_probability": round(ml_probability, 4) if ml_probability is not None else None,
        "method": method,
    }


@app.post("/api/fake-listings/train")
def train_fake_listing_model(request: FakeTrainingRequest) -> dict[str, Any]:
    global fake_listing_model
    labels = [int(example.is_fake) for example in request.examples]
    if len(set(labels)) < 2 or min(labels.count(0), labels.count(1)) < 3:
        raise HTTPException(status_code=422, detail="Training data must include at least 3 real and 3 fake listings.")
    x = [fake_features(example.listing) for example in request.examples]
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(x, labels)
    fake_listing_model = model
    return {"status": "trained_in_memory", "examples": len(labels), "labels": {"real": labels.count(0), "fake": labels.count(1)}}


@app.post("/api/trust/score")
def score_trust(profile: TrustProfile) -> dict[str, Any]:
    rule_score, reasons = trust_rule_score(profile)
    ml_probability = None
    if trust_model is not None:
        ml_probability = float(trust_model.predict_proba([trust_features(profile)])[0][1])
        score = round(0.5 * rule_score + 0.5 * 100 * ml_probability)
        method = "trained Scikit-learn model + rule-based score"
    else:
        score = rule_score
        method = "rule-based score; train the ML model with labeled profiles"
    return {
        "trust_score": score,
        "reasons": reasons,
        "model_trusted_probability": round(ml_probability, 4) if ml_probability is not None else None,
        "method": method,
    }


@app.post("/api/trust/train")
def train_trust_model(request: TrustTrainingRequest) -> dict[str, Any]:
    global trust_model
    labels = [int(example.is_trusted) for example in request.examples]
    if len(set(labels)) < 2 or min(labels.count(0), labels.count(1)) < 3:
        raise HTTPException(status_code=422, detail="Training data must include at least 3 trusted and 3 untrusted profiles.")
    x = [trust_features(example.profile) for example in request.examples]
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(x, labels)
    trust_model = model
    return {"status": "trained_in_memory", "examples": len(labels), "labels": {"untrusted": labels.count(0), "trusted": labels.count(1)}}


@app.post("/api/demand/forecast")
def forecast_demand(request: DemandForecastRequest) -> dict[str, Any]:
    history = sorted(request.history, key=lambda point: point.month)
    for point in history:
        parse_month(point.month)
    counts = [float(point.count) for point in history]
    if len(counts) < 8:
        baseline = float(np.mean(counts[-3:]))
        return {
            "category": request.category,
            "method": "moving-average preview; Random Forest needs at least 8 historical monthly points",
            "trained": False,
            "forecasts": [
                {"month": month_after(history[-1].month, step), "predicted_demand": round(max(0, baseline), 2)}
                for step in range(1, request.horizon + 1)
            ],
        }
    x: list[list[float]] = []
    y: list[float] = []
    for index in range(2, len(history)):
        month_num = parse_month(history[index].month).month
        x.append([
            math.sin(2 * math.pi * month_num / 12),
            math.cos(2 * math.pi * month_num / 12),
            float(index),
            counts[index - 1],
            counts[index - 2],
        ])
        y.append(counts[index])
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=42)
    model.fit(x, y)
    predictions: list[dict[str, Any]] = []
    evolving = counts[:]
    for step in range(1, request.horizon + 1):
        month = month_after(history[-1].month, step)
        month_num = parse_month(month).month
        features = [[
            math.sin(2 * math.pi * month_num / 12),
            math.cos(2 * math.pi * month_num / 12),
            float(len(evolving)),
            evolving[-1],
            evolving[-2],
        ]]
        predicted = max(0.0, float(model.predict(features)[0]))
        predictions.append({"month": month, "predicted_demand": round(predicted, 2)})
        evolving.append(predicted)
    return {"category": request.category, "method": "RandomForestRegressor", "trained": True, "forecasts": predictions}


def preference_score(pet: PetCandidate, preference: PreferenceProfile) -> int:
    score = 40
    indoor_friendly = pet.category in {"kitten", "rabbit", "bird"}
    score += (20 if indoor_friendly else 5) if preference.space == "apartment" else (20 if pet.category == "puppy" else 10)
    score += (20 if pet.category == "puppy" else 10) if preference.activity == "high" else (20 if pet.category in {"kitten", "rabbit"} else 12)
    score += (15 if pet.category == "puppy" else 10) if preference.experience == "experienced" else (15 if pet.category in {"kitten", "rabbit"} else 8)
    score += (15 if pet.category in {"rabbit", "bird"} else 5) if preference.budget == "low" else 10
    return min(100, score)


@app.post("/api/recommendations")
def recommend(request: RecommendationRequest) -> dict[str, Any]:
    interactions = request.interactions
    if not interactions:
        with db_connect() as conn:
            stored = conn.execute("SELECT user_id, item_id, rating FROM interactions").fetchall()
        interactions = [Interaction(**dict(row)) for row in stored]
    user_scores: dict[str, dict[str, float]] = {}
    for interaction in interactions:
        user_scores.setdefault(interaction.user_id, {})[interaction.item_id] = interaction.rating
    cf_scores: dict[str, float] = {}
    if request.user_id in user_scores and len(user_scores) > 1:
        users = list(user_scores)
        items = sorted({item for ratings in user_scores.values() for item in ratings})
        matrix = np.zeros((len(users), len(items)), dtype=np.float32)
        for row, user in enumerate(users):
            for item, rating in user_scores[user].items():
                matrix[row, items.index(item)] = rating
        similarities = cosine_similarity(matrix)
        target_index = users.index(request.user_id)
        neighbor_indices = [i for i in range(len(users)) if i != target_index and similarities[target_index, i] > 0]
        for item in items:
            item_index = items.index(item)
            numerator = sum(similarities[target_index, i] * matrix[i, item_index] for i in neighbor_indices)
            denominator = sum(similarities[target_index, i] for i in neighbor_indices if matrix[i, item_index] > 0)
            if denominator:
                cf_scores[item] = float(numerator / denominator)
    ranked = []
    for candidate in request.candidates:
        lifestyle_score = preference_score(candidate, request.preferences)
        collaborative_score = cf_scores.get(str(candidate.id))
        if collaborative_score is None:
            combined = lifestyle_score
        else:
            combined = round(0.65 * (collaborative_score / 5 * 100) + 0.35 * lifestyle_score)
        ranked.append({"pet": candidate.model_dump(), "score": round(combined, 1), "collaborative_score": collaborative_score})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    used_cf = bool(cf_scores)
    return {
        "method": "user-item collaborative filtering + lifestyle fallback" if used_cf else "lifestyle fallback; add adopter interaction history for collaborative filtering",
        "collaborative_filtering_active": used_cf,
        "results": ranked[: request.limit],
    }


@app.post("/api/adoption/readiness")
def adoption_readiness(profile: AdoptionReadinessRequest) -> dict[str, Any]:
    score = 0
    next_steps: list[str] = []
    factors = [
        (profile.household_agrees, 20, "Talk through the daily care plan with everyone at home."),
        (profile.daily_time != "under_1", 18 if profile.daily_time == "2_plus" else 12, "Set aside predictable daily time for care, play, and settling in."),
        (profile.monthly_budget != "under_3000", 15 if profile.monthly_budget == "over_6000" else 10, "Review recurring costs and leave room for unexpected needs."),
        (profile.home_type != "renter" or profile.landlord_approval, 15, "Confirm your housing rules allow the pet you have in mind."),
        (profile.emergency_fund, 12, "Build an emergency fund for unexpected expenses."),
        (profile.primary_caregiver, 10, "Choose a primary caregiver and agree on backup coverage."),
        (profile.travel_care_plan, 10, "Make a plan for care during travel or busy weeks."),
    ]
    for complete, points, suggestion in factors:
        if complete:
            score += points
        else:
            next_steps.append(suggestion)
    score = min(100, score)
    status = "Ready to plan visits" if score >= 80 else "Good progress" if score >= 60 else "Build your care plan"
    return {"score": score, "status": status, "next_steps": next_steps, "is_approval": False}


@app.post("/api/adoption/welcome-plan")
def adoption_welcome_plan(request: WelcomePlanRequest) -> dict[str, Any]:
    supplies = {
        "puppy": ["Leash and harness", "Safe resting area", "Chew-safe enrichment", "Waste bags"],
        "kitten": ["Litter area", "Scratching surface", "Carrier", "Window and cable safety check"],
        "rabbit": ["Secure exercise pen", "Hiding place", "Non-slip floor area", "Chew-safe enrichment"],
        "bird": ["Secure enclosure", "Perches", "Quiet resting area", "Safe enrichment"],
    }
    focus_by_category = {
        "puppy": "short, calm introductions and a predictable outdoor routine",
        "kitten": "a quiet base room, safe hiding spots, and gradual exploration",
        "rabbit": "a secure floor-level base area and supervised exploration",
        "bird": "a calm enclosure location and unhurried time to settle",
    }
    days = [
        {"day": 1, "title": "Settle in", "tasks": ["Prepare one quiet base area.", "Keep the first day low-key and let the pet approach at its own pace."]},
        {"day": 2, "title": "Build a routine", "tasks": ["Set consistent times for meals and attention.", "Keep water and the pet's core supplies easy to reach."]},
        {"day": 3, "title": "Explore safely", "tasks": ["Open one additional safe area at a time.", "Notice where the pet prefers to rest and retreat."]},
        {"day": 4, "title": "Meet the household", "tasks": ["Introduce household members one at a time.", "Keep interactions calm and brief."]},
        {"day": 5, "title": "Add enrichment", "tasks": ["Offer a safe activity suited to the pet.", "Keep a few minutes for positive one-to-one interaction."]},
        {"day": 6, "title": "Review the setup", "tasks": ["Check that the pet can reach its resting and care areas.", "Adjust the home setup based on what you have observed."]},
        {"day": 7, "title": "Keep what works", "tasks": ["Review the first week's routine.", "Keep a short list of questions for the adoption organization or veterinarian."]},
    ]
    return {
        "pet_name": request.pet_name,
        "category": request.category,
        "first_week_focus": focus_by_category[request.category],
        "setup_checklist": supplies[request.category],
        "days": days,
        "method": "category-aware welcome planner",
    }


@app.post("/api/interactions")
def record_interaction(interaction: InteractionWrite) -> dict[str, str]:
    with db_connect() as conn:
        conn.execute(
            """INSERT INTO interactions (user_id, item_id, rating, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, item_id) DO UPDATE SET
                rating = excluded.rating,
                updated_at = excluded.updated_at""",
            (interaction.user_id, interaction.item_id, interaction.rating, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        )
    return {"status": "recorded"}


@app.get("/api/listings")
def list_created_listings(limit: int = 100) -> dict[str, Any]:
    with db_connect() as conn:
        rows = conn.execute("SELECT * FROM listings ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
    listings = []
    for row in rows:
        listing = dict(row)
        listing["images"] = json.loads(listing.pop("images_json"))
        listing["requires_review"] = bool(listing["requires_review"])
        listings.append(listing)
    return {"listings": listings}


@app.post("/api/media/upload")
async def upload_listing_photo(image: UploadFile = File(...)) -> dict[str, str]:
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB upload limit.")
    try:
        with Image.open(io.BytesIO(data)) as picture:
            picture.verify()
            image_format = picture.format
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=415, detail="Upload a valid JPEG, PNG, or WebP image.") from exc
    extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    extension = extensions.get(image_format or "")
    if not extension:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported.")
    filename = f"{uuid4().hex}{extension}"
    (UPLOAD_DIR / filename).write_bytes(data)
    return {"url": f"/media/{filename}"}


@app.post("/api/listings")
def create_listing(listing: ListingCreate) -> dict[str, Any]:
    with db_connect() as conn:
        cursor = conn.execute(
            """INSERT INTO listings
            (name, category, breed, age, price, location, description, images_json, trust_score, risk_score, requires_review, created_at, sex, food, breed_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                listing.name,
                listing.category,
                listing.breed,
                listing.age,
                listing.price,
                listing.location,
                listing.description,
                json.dumps(listing.images),
                listing.trust_score,
                listing.risk_score,
                int(listing.requires_review),
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                listing.sex,
                listing.food,
                listing.breed_details,
            ),
        )
        listing_id = cursor.lastrowid
    return {"id": listing_id, "status": "saved", "requires_review": listing.requires_review}

