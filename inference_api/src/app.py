import uvicorn

from routes import views
from fastapi import FastAPI
from core.get_configs import InferenceConfig
from starlette.middleware.cors import CORSMiddleware
from apis.inferences.style_transfer import StyleTransfer
from apis.inferences.style_transfer_interpolate import StyleTransferInterpolate


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.InferenceConfig = InferenceConfig()

app.state.StyleTransfer = StyleTransfer(
    app.state.InferenceConfig.load_configuration("baseline.json"))

app.state.StyleTransferInterpolate = StyleTransferInterpolate(
    app.state.InferenceConfig.load_configuration("baseline.json"))

app.include_router(views.router)

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=5000)
