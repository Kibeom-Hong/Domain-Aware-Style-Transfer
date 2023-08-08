import io
import cv2
import numpy as np

from PIL import Image
from fastapi import APIRouter
from starlette.requests import Request
from fastapi.responses import Response
from fastapi import APIRouter, UploadFile, File


router = APIRouter()


def decode_image(file: UploadFile) -> np.ndarray:
    file = file.file.read()
    frame = np.fromstring(file, np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)


def encode_image(image: Image) -> bytes:
    bytes_image = io.BytesIO()
    image.save(bytes_image, format='jpeg')
    return bytes_image


def main(config_name: str, request: Request, content_file: UploadFile = File("content_file"), style_file: UploadFile = File("style_file")):
    content_frame = decode_image(content_file)
    style_frame = decode_image(style_file)
    dependencies = request.app.state

    if config_name == "vanilla" or config_name == "adversarial_loss":
        return dependencies.StyleTransfer.main(
            content_frame, style_frame, dependencies.InferenceConfig.load_configuration(f"{config_name}.json"))
    else:
        return dependencies.StyleTransferInterpolate.main(
            content_frame, style_frame, dependencies.InferenceConfig.load_configuration(f"{config_name}.json"))


@router.post("/style_transfer/vanilla", response_class=Response, tags=["vanilla"])
def vanilla(request: Request, content_file: UploadFile = File("content_file"), style_file: UploadFile = File("style_file")):
    result = encode_image(main(config_name="vanilla", request=request,
                               content_file=content_file, style_file=style_file))
    return Response(content=result.getvalue(), media_type="image/jpeg")


@router.post("/style_transfer/adversarial_loss", response_class=Response, tags=["adversarial_loss"])
def adversarial(request: Request, content_file: UploadFile = File("content_file"), style_file: UploadFile = File("style_file")):
    result = encode_image(main(config_name="adversarial_loss", request=request,
                               content_file=content_file, style_file=style_file))
    return Response(content=result.getvalue(), media_type="image/jpeg")


@router.post("/style_transfer/interpolate", response_class=Response, tags=["interpolate"])
async def interpolate(request: Request, content_file: UploadFile = File("content_file"), style_file: UploadFile = File("style_file")):
    result = encode_image(main(config_name="interpolate", request=request,
                               content_file=content_file, style_file=style_file))
    return Response(content=result.getvalue(), media_type="image/jpeg")
