# Pet Fur-Ever Python backend

FastAPI service for listings, visual search, safety scoring, recommendations, demand forecasts, and adoption planning. Pet profiles store sex, age, breed notes, and current food details.

## Run locally

Requires Python 3.10 or later. Open PowerShell in the extracted `Pet-Fur-Ever-AI-Project` folder:

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-clip.txt
uvicorn app:app --reload --port 8000
```

The CLIP install is required for photo search. The first photo search downloads the model weights (`openai/clip-vit-base-patch32`) and needs internet access. Open `http://127.0.0.1:8000/docs` for the API reference. SQLite data is stored at `backend/data/listings.sqlite3`; existing databases are upgraded with the new profile fields on startup.

In a second PowerShell window, return to the extracted project folder and serve the website:

```powershell
cd C:\path\to\Pet-Fur-Ever-AI-Project
python -m http.server 5500
```

Open `http://127.0.0.1:5500`. Keep both terminal windows running. The logo and favicon are included beside `index.html`. To point the page at a different backend host, add this before the page's main script:

```html
<script>window.PET_FUREVER_API = "http://127.0.0.1:8000";</script>
```

The default CORS list permits common local development origins. Set `PET_CORS_ORIGINS` to a comma-separated list of the exact deployed site origins before hosting the API.

## Upload the project to GitHub

Install Git for Windows and create a new empty repository on GitHub. In PowerShell, open the extracted `Pet-Fur-Ever-AI-Project` folder (the folder containing `index.html`) and run:

```powershell
git --version
git init
git add .
git commit -m "Add Pet Fur-Ever marketplace"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/pet-fur-ever.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use your repository URL. The `.gitignore` file skips the virtual environment, generated database, uploaded images, and local secrets. Create the GitHub repository without an initial README, license, or `.gitignore` so the first push is straightforward.

## Feature endpoints

| Feature | Endpoint | Behavior |
| --- | --- | --- |
| Photo search | `POST /api/vision/clip-search` | Compares the query with actual listing photos using CLIP image embeddings, identifies likely pet category, filters weak matches, and returns up to three close matches. Requires `requirements-clip.txt`. |
| Fake listing detection | `POST /api/fake-listings/score` | Explainable rule score by default; combines rules with a trained Random Forest when available. |
| Fake detector training | `POST /api/fake-listings/train` | Train in-memory from labeled real/fake listing examples. Requires at least 3 examples of each label. |
| Demand forecast | `POST /api/demand/forecast` | Random Forest forecast when given 8+ monthly points; shorter history returns a moving-average preview. |
| Recommendation engine | `POST /api/recommendations` | Collaborative filtering from the supplied user-item ratings, with a lifestyle-fit fallback. |
| Adoption readiness | `POST /api/adoption/readiness` | Score a home and care plan, then return practical next steps; it is not an adoption approval decision. |
| First-week planner | `POST /api/adoption/welcome-plan` | Return a category-aware seven-day home transition checklist. |
| Student trust score | `POST /api/trust/score` | Transparent rule score by default; combines rules with a trained Random Forest when available. |
| Trust model training | `POST /api/trust/train` | Train in-memory from labeled trusted/untrusted profiles. Requires at least 3 examples of each label. |
| Recommendation feedback | `POST /api/interactions` | Store wishlist and adoption ratings in SQLite for collaborative filtering. |
| Listing photos | `POST /api/media/upload` | Validate and store JPEG, PNG, or WebP photos; returns a URL served by the API. |
| Listings | `GET/POST /api/listings` | Read and save listing records in SQLite, including sex, age, breed details, current food, photos, and review signals. |
| API status | `GET /api/health` | Reports service and model readiness. |

Training endpoints keep models in memory. Restarting the service clears the trained models, so persist approved training data and model artifacts before production deployment. The service deliberately does not invent training labels: submit representative, reviewed examples before relying on trained scores.

Photo search sends compressed listing photos from the browser for model comparison. Remote image hosts must allow cross-origin photo access for their photos to be searchable. If no photos pass the similarity threshold, the endpoint returns no matches instead of the whole catalogue.

To train the fake-listing classifier, send reviewed records to `POST /api/fake-listings/train`. Use the same `listing` fields accepted by `/api/fake-listings/score`, and provide an `is_fake` label for each record. Include at least 3 examples of each label. The trust classifier follows the same pattern at `/api/trust/train`, with `profile` fields from `/api/trust/score` and an `is_trusted` label.

## Example requests

Safety score:

```json
POST /api/fake-listings/score
{
  "name": "Milo",
  "category": "puppy",
  "breed": "Indie dog",
  "age": "4 months",
  "price": "₹0 / adoption",
  "description": "Vaccinated and checked by our local veterinarian. Meet Milo in person.",
  "photo_count": 1
}
```

Demand forecast:

```json
POST /api/demand/forecast
{
  "category": "puppy",
  "horizon": 3,
  "history": [
    {"month": "2025-01", "count": 12},
    {"month": "2025-02", "count": 15},
    {"month": "2025-03", "count": 18},
    {"month": "2025-04", "count": 14},
    {"month": "2025-05", "count": 17},
    {"month": "2025-06", "count": 20},
    {"month": "2025-07", "count": 22},
    {"month": "2025-08", "count": 19}
  ]
}
```

For a deployed site, also add authentication and authorization around listing creation, scoring, model training, and trust-profile inputs. Student and identity verification values must come from a verification workflow, not checkboxes or untrusted browser fields.
