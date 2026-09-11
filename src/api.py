from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.production_inference import ProductionInferenceEngine
from src.production_localization import ProductionLocalizer


engine = None
localizer = None

MAX_IMAGE_PIXELS = 50_000_000
WARNING_IMAGE_PIXELS = 25_000_000


class InspectionResponse(BaseModel):
    category: str
    l4_score: float
    l8_score: float
    anomaly_score: float
    threshold: float
    review_threshold: float
    decision: str


class BoundingBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int
    x2: int
    y2: int


class CenterResponse(BaseModel):
    x: float
    y: float
    x_normalized: float
    y_normalized: float


class RegionStatisticsResponse(BaseModel):
    pixel_count: int
    mean_anomaly: float
    max_anomaly: float
    min_anomaly: float


class LocalizationResponse(BaseModel):
    method: str
    bounding_box: BoundingBoxResponse | None
    center: CenterResponse | None
    region_statistics: RegionStatisticsResponse | None


class APIResponse(BaseModel):
    status: str
    filename: str
    result: InspectionResponse
    localization: LocalizationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global localizer

    print("Loading VISYN production engine...")

    engine = ProductionInferenceEngine()
    localizer = ProductionLocalizer(engine)

    print("VISYN production engine loaded.")

    yield

    engine = None
    localizer = None

    print("VISYN production engine unloaded.")


app = FastAPI(
    title="VISYN",
    description="Industrial visual anomaly detection API.",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://visyn-chi.vercel.app",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if engine is None:
        return {
            "status": "starting",
            "engine_loaded": False,
        }

    return {
        "status": "healthy",
        "engine_loaded": True,
        "categories": sorted(engine.reference_bank_paths.keys()),
    }


@app.post(
    "/inspect",
    response_model=APIResponse,
)
async def inspect(
    category: str,
    file: UploadFile = File(...),
):
    if engine is None or localizer is None:
        raise HTTPException(
            status_code=503,
            detail="Production engine is not loaded.",
        )

    if category not in engine.reference_bank_paths:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported category: {category}. "
                f"Supported categories: "
                f"{sorted(engine.reference_bank_paths.keys())}"
            ),
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPEG, PNG, or WebP.",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            width, height = image.size
            pixel_count = width * height

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image.",
        ) from exc

    if pixel_count > MAX_IMAGE_PIXELS:
        megapixels = pixel_count / 1_000_000

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. Maximum supported size is "
                f"50 megapixels. This image is {megapixels:.1f} "
                "megapixels."
            ),
        )

    temporary_path = None

    try:
        with NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        raw_result = engine.inspect(
            category=category,
            image_path=temporary_path,
        )

        localization_result = localizer.localize(
            image_path=temporary_path,
            category=category,
        )

        result = InspectionResponse(
            category=raw_result["category"],
            l4_score=raw_result["l4_score"],
            l8_score=raw_result["l8_score"],
            anomaly_score=raw_result["anomaly_score"],
            threshold=raw_result["threshold"],
            review_threshold=raw_result["review_threshold"],
            decision=raw_result["decision"],
        )

        bounding_box = localization_result.get("bounding_box")
        center = localization_result.get("center")
        region_statistics = localization_result.get(
            "region_statistics"
        )

        localization = LocalizationResponse(
            method=localization_result["method"],
            bounding_box=(
                BoundingBoxResponse(
                    x=bounding_box["x"],
                    y=bounding_box["y"],
                    width=bounding_box["width"],
                    height=bounding_box["height"],
                    x2=bounding_box["x2"],
                    y2=bounding_box["y2"],
                )
                if bounding_box is not None
                else None
            ),
            center=(
                CenterResponse(
                    x=center["x"],
                    y=center["y"],
                    x_normalized=center["x_normalized"],
                    y_normalized=center["y_normalized"],
                )
                if center is not None
                else None
            ),
            region_statistics=(
                RegionStatisticsResponse(
                    pixel_count=region_statistics["pixel_count"],
                    mean_anomaly=region_statistics["mean_anomaly"],
                    max_anomaly=region_statistics["max_anomaly"],
                    min_anomaly=region_statistics["min_anomaly"],
                )
                if region_statistics is not None
                else None
            ),
        )

        return APIResponse(
            status="success",
            filename=file.filename,
            result=result,
            localization=localization,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inspection failed: {exc}",
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)