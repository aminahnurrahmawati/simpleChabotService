import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "core:app",  #core = nama folder, app : pemanggilan Fast api ada di file __init__.py
        host="0.0.0.0",
        port=5000,
        reload=True
    )