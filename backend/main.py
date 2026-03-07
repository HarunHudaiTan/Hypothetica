"""Entry point for the Hypothetica API server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8005, reload=True)
