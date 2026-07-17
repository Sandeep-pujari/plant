import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  X, 
  AlertCircle, 
  Zap, 
  Percent, 
  FileText,
  Activity,
  CheckCircle,
  XCircle,
  HelpCircle,
  Leaf
} from 'lucide-react';
import { Chart, DoughnutController, ArcElement, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
Chart.register(DoughnutController, ArcElement, Tooltip, Legend);

// Self-contained chart component to handle Chart.js canvas life cycle
function PurityChart({ pureVal, impureVal, isUnknown }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    const labels = isUnknown ? ["Unknown Specimen"] : ["Pure Confidence", "Impure Confidence"];
    const data = isUnknown ? [100] : [pureVal, impureVal];
    const colors = isUnknown ? ["#f59e0b"] : ["#10b981", "#f43f5e"];

    chartRef.current = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: colors,
          borderColor: "#121824", // matches glassmorphic card bg
          borderWidth: 3,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "75%",
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                if (isUnknown) {
                  return ` ${context.label}: Non-plant or Unknown`;
                }
                return ` ${context.label}: ${Number(context.raw).toFixed(2)}%`;
              }
            }
          }
        }
      }
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [pureVal, impureVal, isUnknown]);

  return <canvas ref={canvasRef} />;
}

function App() {
  const [activeFile, setActiveFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'webp'];
  const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB

  // Cleanup preview URL object on unmount or file change
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelection = (file) => {
    setError(null);
    setResult(null);

    if (!file) return;

    const fileExt = file.name.split('.').pop().toLowerCase();
    
    // 1. Validate File Format
    if (!ALLOWED_EXTENSIONS.includes(fileExt)) {
      setError(`Unsupported file format. Supported extensions: ${ALLOWED_EXTENSIONS.join(', ')}`);
      clearFileState();
      return;
    }

    // 2. Validate File Size
    if (file.size > MAX_FILE_SIZE) {
      setError("File size exceeds the 16MB limit.");
      clearFileState();
      return;
    }

    // Set active file and create blob URL for preview
    setActiveFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const clearFileState = () => {
    setActiveFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Drag and drop event handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const triggerFileSelect = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  // Run Flask model inference API
  const analyzePlantSpecimen = async () => {
    if (!activeFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", activeFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server returned status ${response.status}`);
      }

      if (data.success) {
        setResult(data);
        // Smooth scroll to results after a short delay for rendering
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
      } else {
        throw new Error(data.error || "Inference pipeline failed.");
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "An unexpected error occurred during prediction.");
    } finally {
      setIsLoading(false);
    }
  };

  // Determine styling states based on result purity/class
  const isUnknown = result ? (result.class === "UNKNOWN" || result.purity.toLowerCase().includes("unknown")) : false;
  const isPure = result ? (!isUnknown && result.purity.toLowerCase().startsWith("pure")) : false;

  let resultCardClass = "card results-card";
  let PurityIcon = HelpCircle;
  
  if (result) {
    if (isUnknown) {
      resultCardClass += " status-unknown";
      PurityIcon = HelpCircle;
    } else if (isPure) {
      resultCardClass += " status-pure";
      PurityIcon = CheckCircle;
    } else {
      resultCardClass += " status-impure";
      PurityIcon = XCircle;
    }
  }

  // Extract raw probabilities
  const femaleProb = result?.probabilities?.female || 0;
  const hybridProb = result?.probabilities?.hybrid || 0;
  const maleProb = result?.probabilities?.male || 0;

  // Doughnut chart calculation
  const chartPureVal = hybridProb;
  const chartImpureVal = femaleProb + maleProb;

  return (
    <div className="app-container">
      {/* Header Section */}
      <header className="app-header">
        <span className="project-tag">
          <span className="project-tag-pulse"></span>
          Computer Vision Inference
        </span>
        <h1 className="main-title">Plant Genetic Purity Analyzer</h1>
        <p className="subtitle">
          AI-assisted genetic purity prediction using morphological analysis of germinated parent lines and target hybrids.
        </p>
      </header>

      {/* Main Workspace */}
      <main style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
        {/* Upload Card */}
        <section className="card">
          <div className="card-header">
            <h2>Upload Plant Specimen</h2>
            <p className="card-desc">Select or drag an image of a germinated plant leaf/stem for analysis.</p>
          </div>
          
          <div 
            className={`drop-zone ${isDragging ? 'drag-over' : ''}`}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={!activeFile ? triggerFileSelect : undefined}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              accept="image/png, image/jpeg, image/jpg, image/bmp, image/webp" 
              style={{ display: 'none' }}
              onChange={handleInputChange}
            />
            
            {!previewUrl ? (
              <div className="upload-prompt">
                <div className="upload-icon">
                  <Upload size={36} />
                </div>
                <p className="prompt-text">Drag and drop plant image here</p>
                <span className="prompt-or">or</span>
                <button 
                  type="button" 
                  className="btn btn-secondary"
                  onClick={(e) => {
                    e.stopPropagation();
                    triggerFileSelect();
                  }}
                >
                  Browse Image
                </button>
                <p className="prompt-subtext">Supports PNG, JPG, JPEG, BMP, WEBP up to 16MB</p>
              </div>
            ) : (
              <div className="preview-container" onClick={(e) => e.stopPropagation()}>
                <img src={previewUrl} alt="Plant specimen preview" />
                <button 
                  type="button" 
                  className="remove-btn" 
                  title="Remove image"
                  onClick={clearFileState}
                >
                  <X size={20} />
                </button>
              </div>
            )}
          </div>

          {/* Error alert message */}
          {error && (
            <div className="error-alert">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {/* Analyze Button */}
          {activeFile && (
            <div className="action-area">
              <button 
                type="button" 
                className="btn btn-primary btn-block"
                onClick={analyzePlantSpecimen}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <div className="spinner"></div>
                    <span>Analyzing Specimen...</span>
                  </>
                ) : (
                  <>
                    <Leaf size={18} />
                    <span>Analyze Plant Specimen</span>
                  </>
                )}
              </button>
            </div>
          )}
        </section>

        {/* Results Diagnostic Card */}
        {result && (
          <section ref={resultsRef} className={resultCardClass}>
            <div className="card-header">
              <div className="results-header-flex">
                <div>
                  <h2>Analysis Diagnostics</h2>
                  <p className="card-desc">Comprehensive genetic purity and class breakdown reports.</p>
                </div>
                <div className="status-badge">
                  <span className="status-badge-dot"></span>
                  <PurityIcon size={16} style={{ marginRight: '4px' }} />
                  <span>{result.purity}</span>
                </div>
              </div>
            </div>

            <div className="results-grid">
              {/* Left Column: Doughnut Chart */}
              <div className="chart-column">
                <div className="chart-wrapper">
                  <PurityChart 
                    pureVal={chartPureVal} 
                    impureVal={chartImpureVal} 
                    isUnknown={isUnknown} 
                  />
                  <div className="chart-center-value">
                    <span className="chart-center-val">
                      {isUnknown ? "N/A" : `${chartPureVal.toFixed(1)}%`}
                    </span>
                    <span className="chart-center-lbl">
                      {isUnknown ? "Unknown" : "Pure"}
                    </span>
                  </div>
                </div>
                
                <div className="chart-legend-box">
                  <div className="legend-item">
                    <span className="dot dot-pure"></span>
                    <span className="legend-label">Pure Confidence</span>
                    <span className="legend-val">
                      {isUnknown ? "0.00%" : `${chartPureVal.toFixed(2)}%`}
                    </span>
                  </div>
                  <div className="legend-item">
                    <span className="dot dot-impure"></span>
                    <span className="legend-label">Impure Confidence</span>
                    <span className="legend-val">
                      {isUnknown ? "0.00%" : `${chartImpureVal.toFixed(2)}%`}
                    </span>
                  </div>
                </div>
              </div>

              {/* Right Column: Key Details & Class Distributions */}
              <div className="details-column">
                {/* Key Metrics Grid */}
                <div className="metrics-grid">
                  <div className="metric-tile">
                    <div className="metric-label">Predicted Class</div>
                    <div className="metric-value">{result.class}</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label flex-row" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Percent size={10} />
                      <span>Confidence</span>
                    </div>
                    <div className="metric-value">{result.confidence}</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label flex-row" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Zap size={10} />
                      <span>Inference Speed</span>
                    </div>
                    <div className="metric-value">{result.prediction_time}</div>
                  </div>
                </div>

                {/* Class Probabilities Progress Bars */}
                <div className="probabilities-section">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Activity size={14} className="text-secondary" />
                    <h3>Morphological Class Distributions</h3>
                  </div>
                  
                  {/* Female Row */}
                  <div className="prob-row">
                    <div className="prob-labels">
                      <span className="class-name">Female Specimen</span>
                      <span className="class-percentage">{femaleProb.toFixed(2)}%</span>
                    </div>
                    <div className="progress-track">
                      <div 
                        className="progress-bar female" 
                        style={{ width: `${femaleProb}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Hybrid Row */}
                  <div className="prob-row">
                    <div className="prob-labels">
                      <span className="class-name">Hybrid Target</span>
                      <span className="class-percentage">{hybridProb.toFixed(2)}%</span>
                    </div>
                    <div className="progress-track">
                      <div 
                        className="progress-bar hybrid" 
                        style={{ width: `${hybridProb}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Male Row */}
                  <div className="prob-row">
                    <div className="prob-labels">
                      <span className="class-name">Male Specimen</span>
                      <span className="class-percentage">{maleProb.toFixed(2)}%</span>
                    </div>
                    <div className="progress-track">
                      <div 
                        className="progress-bar male" 
                        style={{ width: `${maleProb}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Decision Rationale */}
                <div className="decision-reason-box">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <FileText size={14} className="text-secondary" />
                    <h4>Decision Rationale</h4>
                  </div>
                  <p className="reason-text">{result.reason || "No detail logs available."}</p>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer-bar">
        <p>&copy; {new Date().getFullYear()} Plant Genetics Classification Project. Headless inference engine powered by MobileNet backend.</p>
      </footer>
    </div>
  );
}

export default App;
