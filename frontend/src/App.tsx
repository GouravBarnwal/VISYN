import { useEffect, useRef, useState } from "react";
import "./App.css";

const categories = [
  { value: "bottle", label: "Bottle" },
  { value: "cable", label: "Cable" },
  { value: "capsule", label: "Capsule" },
  { value: "hazelnut", label: "Hazelnut" },
  { value: "metal_nut", label: "Metal Nut" },
  { value: "screw", label: "Screw" },
];

const acceptedTypes = ["image/png", "image/jpeg"];

const MAX_IMAGE_PIXELS = 50_000_000;
const WARNING_IMAGE_PIXELS = 25_000_000;

type InspectionResult = {
  decision: "PASS" | "REVIEW" | "FAIL";
  anomalyScore: number;
  threshold: number;
  reviewThreshold: number;
  l4Score: number;
  l8Score: number;
};

type LocalizationResult = {
  method: string;
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number;
    x2: number;
    y2: number;
  } | null;
  center: {
    x: number;
    y: number;
    x_normalized: number;
    y_normalized: number;
  } | null;
  region_statistics: {
    pixel_count: number;
    mean_anomaly: number | null;
    max_anomaly: number | null;
    min_anomaly: number | null;
  } | null;
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [category, setCategory] = useState("bottle");
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");
  const [inspectionReady, setInspectionReady] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);

  const [localization, setLocalization] = useState<LocalizationResult | null>(
    null,
  );

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });

  /*
   * This represents the ACTUAL visible image rectangle inside
   * the preview wrapper after object-fit: contain is applied.
   *
   * It is intentionally different from the <img> element's
   * bounding rectangle.
   */
  const [renderedImage, setRenderedImage] = useState({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });

  const [inspectionResult, setInspectionResult] =
    useState<InspectionResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const previewImageWrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;

    const checkBackend = async () => {
      try {
        const response = await fetch("https://visyn.onrender.com/health", {
          method: "GET",
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error("Backend health check failed");
        }

        const data = await response.json();

        if (active) {
          setBackendOnline(
            data.status === "healthy" && data.engine_loaded === true,
          );
        }
      } catch {
        if (active) {
          setBackendOnline(false);
        }
      }
    };

    checkBackend();

    const interval = window.setInterval(checkBackend, 10000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  /*
   * Calculate the exact rectangle occupied by the image
   * after CSS object-fit: contain.
   *
   * The backend bounding box is expressed in ORIGINAL IMAGE
   * coordinates. Therefore the overlay must use this rectangle,
   * not the full preview wrapper.
   */
  useEffect(() => {
    const wrapper = previewImageWrapRef.current;

    if (!wrapper || imageDimensions.width <= 0 || imageDimensions.height <= 0) {
      setRenderedImage({
        left: 0,
        top: 0,
        width: 0,
        height: 0,
      });

      return;
    }

    const updateRenderedImage = () => {
      const containerWidth = wrapper.clientWidth;
      const containerHeight = wrapper.clientHeight;

      if (containerWidth <= 0 || containerHeight <= 0) {
        return;
      }

      const sourceWidth = imageDimensions.width;
      const sourceHeight = imageDimensions.height;

      const sourceAspectRatio = sourceWidth / sourceHeight;

      const containerAspectRatio = containerWidth / containerHeight;

      let displayedWidth: number;
      let displayedHeight: number;

      if (sourceAspectRatio > containerAspectRatio) {
        /*
         * Image is wider than the container.
         * Width fills the container and height is scaled.
         */
        displayedWidth = containerWidth;
        displayedHeight = containerWidth / sourceAspectRatio;
      } else {
        /*
         * Image is taller than the container.
         * Height fills the container and width is scaled.
         */
        displayedHeight = containerHeight;
        displayedWidth = containerHeight * sourceAspectRatio;
      }

      const left = (containerWidth - displayedWidth) / 2;

      const top = (containerHeight - displayedHeight) / 2;

      setRenderedImage({
        left,
        top,
        width: displayedWidth,
        height: displayedHeight,
      });
    };

    updateRenderedImage();

    const observer = new ResizeObserver(updateRenderedImage);

    observer.observe(wrapper);

    window.addEventListener("resize", updateRenderedImage);

    return () => {
      observer.disconnect();

      window.removeEventListener("resize", updateRenderedImage);
    };
  }, [imageDimensions, previewUrl]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const loadImageFile = (file: File) => {
    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result !== "string") {
        setError("Unable to read the selected image.");
        return;
      }

      const imageDataUrl = reader.result;
      const image = new Image();

      image.onload = () => {
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        const pixelCount = width * height;
        const megapixels = pixelCount / 1_000_000;

        if (pixelCount > MAX_IMAGE_PIXELS) {
          setSelectedFile(null);
          setPreviewUrl(null);

          setImageDimensions({
            width: 0,
            height: 0,
          });

          setRenderedImage({
            left: 0,
            top: 0,
            width: 0,
            height: 0,
          });

          setInspectionReady(false);
          setInspectionResult(null);
          setLocalization(null);

          setError(
            `Image is too large. Maximum supported size is 50 megapixels. This image is ${megapixels.toFixed(1)} megapixels.`,
          );

          if (fileInputRef.current) {
            fileInputRef.current.value = "";
          }

          return;
        }

        if (pixelCount > WARNING_IMAGE_PIXELS) {
          setError(
            `Large image detected (${megapixels.toFixed(1)} megapixels). For best performance, use an image below 25 megapixels.`,
          );
        } else {
          setError("");
        }

        setImageDimensions({
          width,
          height,
        });

        setPreviewUrl(imageDataUrl);
      };

      image.onerror = () => {
        setSelectedFile(null);
        setPreviewUrl(null);

        setImageDimensions({
          width: 0,
          height: 0,
        });

        setRenderedImage({
          left: 0,
          top: 0,
          width: 0,
          height: 0,
        });

        setInspectionReady(false);
        setInspectionResult(null);
        setLocalization(null);

        setError(
          "Unable to load this image. Please select a valid PNG or JPEG image.",
        );

        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      };

      image.src = imageDataUrl;
    };

    reader.onerror = () => {
      setError("Unable to read the selected image.");
    };

    reader.readAsDataURL(file);
  };

  const handleFile = (file: File | undefined) => {
    setError("");
    setInspectionReady(false);
    setInspectionResult(null);
    setLocalization(null);

    if (!file) {
      return;
    }

    if (!acceptedTypes.includes(file.type)) {
      setSelectedFile(null);
      setPreviewUrl(null);

      setImageDimensions({
        width: 0,
        height: 0,
      });

      setRenderedImage({
        left: 0,
        top: 0,
        width: 0,
        height: 0,
      });

      setError("Unsupported file type. Use PNG or JPEG images.");

      return;
    }

    /*
     * Store the file only after basic type validation.
     * loadImageFile performs the pixel-dimension validation
     * before displaying the image.
     */
    setSelectedFile(file);
    loadImageFile(file);
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(event.target.files?.[0]);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);

    handleFile(event.dataTransfer.files[0]);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError("");
    setInspectionReady(false);
    setInspectionResult(null);
    setLocalization(null);

    setImageDimensions({
      width: 0,
      height: 0,
    });

    setRenderedImage({
      left: 0,
      top: 0,
      width: 0,
      height: 0,
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const startCamera = async () => {
    setError("");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not supported by this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
        },
        audio: false,
      });

      streamRef.current = stream;
      setCameraOpen(true);

      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      });
    } catch {
      setCameraOpen(false);

      setError(
        "Unable to access the camera. Check browser permissions and camera availability.",
      );
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOpen(false);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      !video ||
      !canvas ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      setError("Camera image is not ready yet. Please try again.");

      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      setError("Unable to capture the camera image.");
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("Unable to create the captured image.");

          return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");

        const file = new File([blob], `visyn-camera-${timestamp}.jpg`, {
          type: "image/jpeg",
        });

        setSelectedFile(file);
        loadImageFile(file);
        setError("");
        setInspectionReady(false);
        setInspectionResult(null);
        setLocalization(null);

        stopCamera();
      },
      "image/jpeg",
      0.92,
    );
  };

  const handleInspection = async () => {
    if (!selectedFile || isInspecting) {
      return;
    }

    setIsInspecting(true);
    setError("");
    setInspectionReady(false);
    setInspectionResult(null);
    setLocalization(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `https://visyn.onrender.com/inspect?category=${encodeURIComponent(category)}`,
        {
          method: "POST",
          body: formData,
        },
      );

      let data: {
        status?: string;
        result?: {
          decision?: "PASS" | "REVIEW" | "FAIL";
          anomaly_score?: number;
          threshold?: number;
          review_threshold?: number;
          l4_score?: number;
          l8_score?: number;
        };
        localization?: LocalizationResult;
        detail?: string;
      };

      try {
        data = await response.json();
      } catch {
        throw new Error("Backend returned an invalid response.");
      }

      if (!response.ok) {
        throw new Error(
          data.detail || `Inspection failed with HTTP ${response.status}.`,
        );
      }

      if (!data.result) {
        throw new Error(
          "Backend response did not contain an inspection result.",
        );
      }

      const result = data.result;

      const validDecisions = ["PASS", "REVIEW", "FAIL"] as const;

      if (
        !result.decision ||
        !validDecisions.includes(result.decision) ||
        result.anomaly_score === undefined ||
        result.threshold === undefined ||
        result.review_threshold === undefined ||
        result.l4_score === undefined ||
        result.l8_score === undefined
      ) {
        throw new Error(
          "Backend returned an incomplete or invalid inspection result.",
        );
      }

      setInspectionResult({
        decision: result.decision,
        anomalyScore: result.anomaly_score,
        threshold: result.threshold,
        reviewThreshold: result.review_threshold,
        l4Score: result.l4_score,
        l8Score: result.l8_score,
      });

      setLocalization(data.localization ?? null);
      setInspectionReady(true);
    } catch (inspectionError) {
      setError(
        inspectionError instanceof Error
          ? inspectionError.message
          : "Inspection failed. Please try again.",
      );
    } finally {
      setIsInspecting(false);
    }
  };

  const categoryLabel =
    categories.find((item) => item.value === category)?.label ?? category;

  const boundingBox = localization?.bounding_box;

  /*
   * Convert original-image coordinates into the exact
   * displayed image coordinates.
   */
  const localizationStyle =
    boundingBox &&
    inspectionResult &&
    imageDimensions.width > 0 &&
    imageDimensions.height > 0 &&
    renderedImage.width > 0 &&
    renderedImage.height > 0
      ? {
          left: `${
            renderedImage.left +
            (boundingBox.x / imageDimensions.width) * renderedImage.width
          }px`,

          top: `${
            renderedImage.top +
            (boundingBox.y / imageDimensions.height) * renderedImage.height
          }px`,

          width: `${
            (boundingBox.width / imageDimensions.width) * renderedImage.width
          }px`,

          height: `${
            (boundingBox.height / imageDimensions.height) * renderedImage.height
          }px`,
        }
      : undefined;

  return (
    <div className="visyn-app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-symbol">V</div>

          <div className="brand-copy">
            <h1>VISYN</h1>

            <span>Deep Visual Intelligence for Quality Inspection</span>
          </div>
        </div>

        <div
          className={`system-indicator ${
            backendOnline ? "backend-online" : "backend-offline"
          }`}
        >
          <span className="indicator-dot" />

          <span>{backendOnline ? "BACKEND ONLINE" : "BACKEND OFFLINE"}</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div>
            <span className="section-kicker">QUALITY INSPECTION SYSTEM</span>

            <h2>Visual Inspection Console</h2>

            <p>
              Analyze industrial components with VISYN&apos;s production
              computer vision pipeline.
            </p>
          </div>

          <div className="engine-card">
            <span>MODEL</span>

            <strong>MobileNetV3-Small</strong>

            <small>L4 + L8 feature representation</small>
          </div>
        </section>

        <section className="inspection-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <span className="panel-number">01</span>

                <h3>Inspection Input</h3>
              </div>

              <span className="panel-status">
                {selectedFile ? "IMAGE LOADED" : "READY"}
              </span>
            </div>

            <input
              ref={fileInputRef}
              className="hidden-file-input"
              type="file"
              accept=".png,.jpg,.jpeg,image/png,image/jpeg"
              onChange={handleFileInput}
            />

            <canvas ref={canvasRef} className="hidden-file-input" />

            {cameraOpen ? (
              <div className="camera-zone">
                <div className="camera-preview">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="camera-video"
                  />

                  <div className="camera-frame" />
                </div>

                <div className="camera-controls">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={stopCamera}
                  >
                    Cancel
                  </button>

                  <button
                    type="button"
                    className="capture-button"
                    onClick={captureImage}
                  >
                    Capture Image
                  </button>
                </div>
              </div>
            ) : previewUrl && selectedFile ? (
              <div className="preview-zone">
                <div ref={previewImageWrapRef} className="preview-image-wrap">
                  <img
                    src={previewUrl}
                    alt="Selected inspection sample"
                    className="preview-image"
                  />

                  {localizationStyle && (
                    <div className="localization-box" style={localizationStyle}>
                      <span className="localization-label">
                        EVIDENCE REGION
                      </span>
                    </div>
                  )}

                  <button
                    type="button"
                    className="remove-image"
                    onClick={handleClear}
                    aria-label="Remove selected image"
                  >
                    ×
                  </button>
                </div>

                <div className="file-summary">
                  <div>
                    <strong>{selectedFile.name}</strong>

                    <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                  </div>

                  <button
                    type="button"
                    className="change-image"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Change image
                  </button>
                </div>
              </div>
            ) : (
              <div
                className={`upload-zone ${isDragging ? "dragging" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    fileInputRef.current?.click();
                  }
                }}
              >
                <div className="upload-symbol">+</div>

                <h4>Upload inspection image</h4>

                <p>Drag and drop an image here or select a file</p>

                <span className="file-types">PNG · JPG · JPEG</span>

                <div className="input-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={(event) => {
                      event.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                  >
                    Select Image
                  </button>

                  <button
                    type="button"
                    className="camera-button"
                    onClick={(event) => {
                      event.stopPropagation();
                      startCamera();
                    }}
                  >
                    Use Camera
                  </button>
                </div>
              </div>
            )}

            {error && <div className="input-error">{error}</div>}

            <div className="form-field">
              <label htmlFor="category">Component Category</label>

              <select
                id="category"
                value={category}
                onChange={(event) => {
                  setCategory(event.target.value);
                  setInspectionReady(false);
                  setInspectionResult(null);
                  setLocalization(null);
                  setError("");
                }}
              >
                {categories.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="button"
              className={`primary-button ${
                inspectionReady ? "inspection-ready" : ""
              }`}
              disabled={!selectedFile || isInspecting}
              onClick={handleInspection}
            >
              {isInspecting
                ? "Inspecting..."
                : inspectionReady
                  ? "Inspection Ready"
                  : "Run Inspection"}

              <span>{isInspecting ? "..." : inspectionReady ? "✓" : "→"}</span>
            </button>

            {inspectionReady && (
              <div className="inspection-note">
                <span />
                {categoryLabel} sample ready for production inspection.
              </div>
            )}
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-number">02</span>

                <h3>Inspection Result</h3>
              </div>

              <span
                className={`panel-status ${
                  inspectionResult
                    ? `result-${inspectionResult.decision.toLowerCase()}`
                    : "waiting"
                }`}
              >
                {inspectionResult ? inspectionResult.decision : "WAITING"}
              </span>
            </div>

            {inspectionResult ? (
              <div className="result-content">
                <div
                  className={`decision-display decision-${inspectionResult.decision.toLowerCase()}`}
                >
                  <span className="decision-label">FINAL DECISION</span>

                  <strong>
                    {inspectionResult.decision === "FAIL"
                      ? "DEFECTIVE"
                      : inspectionResult.decision}
                  </strong>

                  {inspectionResult.decision === "PASS" && <p>NO DEFECT</p>}

                  {inspectionResult.decision === "REVIEW" && (
                    <p>MANUAL REVIEW REQUIRED</p>
                  )}
                </div>

                <div className="score-display">
                  <span>ANOMALY SCORE</span>

                  <strong>{inspectionResult.anomalyScore.toFixed(4)}</strong>
                </div>

                <div className="result-metrics">
                  <div>
                    <span>THRESHOLD</span>

                    <strong>{inspectionResult.threshold.toFixed(4)}</strong>
                  </div>

                  <div>
                    <span>REVIEW BOUNDARY</span>

                    <strong>
                      {inspectionResult.reviewThreshold.toFixed(4)}
                    </strong>
                  </div>

                  <div>
                    <span>L4 SCORE</span>

                    <strong>{inspectionResult.l4Score.toFixed(4)}</strong>
                  </div>

                  <div>
                    <span>L8 SCORE</span>

                    <strong>{inspectionResult.l8Score.toFixed(4)}</strong>
                  </div>
                </div>

                <div className="result-policy">
                  <span>DECISION POLICY</span>

                  <p>
                    PASS below threshold · REVIEW within review boundary · FAIL
                    above review boundary
                  </p>
                </div>

                {localization && (
                  <div className="localization-summary">
                    <span>LOCALIZATION</span>

                    <p>
                      {boundingBox
                        ? "Suspicious evidence region identified in the inspection image."
                        : "No suspicious evidence region identified."}
                    </p>

                    {boundingBox && (
                      <small>
                        Region: {boundingBox.x}, {boundingBox.y} ·{" "}
                        {boundingBox.width} × {boundingBox.height}
                      </small>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="result-empty">
                <div className="result-mark">
                  <span>V</span>
                </div>

                <h4>No inspection result</h4>

                <p>
                  {inspectionReady
                    ? "The image is ready for inspection."
                    : "Upload an image or capture one with the camera to begin analysis."}
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="system-overview">
          <div className="overview-item">
            <span>DECISION POLICY</span>

            <strong>PASS / REVIEW / FAIL</strong>
          </div>

          <div className="overview-item">
            <span>FEATURE LAYERS</span>

            <strong>L4 + L8</strong>
          </div>

          <div className="overview-item">
            <span>SCORING</span>

            <strong>TOP-5 MEAN</strong>
          </div>

          <div className="overview-item">
            <span>LOCALIZATION</span>

            <strong>P95 EVIDENCE</strong>
          </div>
        </section>

        <section className="pipeline-section">
          <div className="pipeline-heading">
            <span className="section-kicker">SYSTEM PIPELINE</span>

            <h3>Production inspection workflow</h3>
          </div>

          <div className="pipeline">
            <div className="pipeline-step">
              <span>01</span>

              <strong>IMAGE</strong>

              <small>Input acquisition</small>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <span>02</span>

              <strong>FEATURES</strong>

              <small>L4 + L8 extraction</small>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <span>03</span>

              <strong>SCORING</strong>

              <small>Reference comparison</small>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <span>04</span>

              <strong>DECISION</strong>

              <small>PASS / REVIEW / FAIL</small>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>VISYN</span>

        <span>
          Deep Visual Intelligence for Quality Inspection - Built by Gourav
          Barnwal
        </span>

        <span>Production Computer Vision</span>
      </footer>
    </div>
  );
}

export default App;
