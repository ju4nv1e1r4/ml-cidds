from fastapi import FastAPI
import uvicorn
import threading

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

def start_subscriber():
    from workers.sub_new_data import main
    main()

if __name__ == "__main__":
    thread = threading.Thread(target=start_subscriber)
    thread.start()

    uvicorn.run("workers.subscriber:app", host="0.0.0.0", port=8080, log_level="info")
