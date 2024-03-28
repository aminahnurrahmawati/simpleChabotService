from fastapi import FastAPI
from core.routes import service
app = FastAPI(
    title="Testing QA Based on document Service",
    description="Q and A using OpenAI",
    version="0.0.1"
)

app.include_router(
    service.route, #restaurantAPI.router
    prefix='/v1'
)