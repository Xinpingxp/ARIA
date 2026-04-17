import React, { useState, useRef } from 'react';
import './index.css';

function App() {
  const [isScanning, setIsScanning] = useState(false);
  const [results, setResults] = useState(null);
  const [heatmapImage, setHeatmapImage] = useState(null);
  const [demoMode, setDemoMode] = useState(false);
  const [fileName, setFileName] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleUpload = () => {
    fileInputRef.current.click();
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  const processFile = async (file) => {
    setFileName(file.name);
    setIsScanning(true);
    setResults(null);
    setHeatmapImage(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.error) throw new Error(data.error);

      setResults({
        cn: data.probabilities.cn,
        mci: data.probabilities.mci,
        ad: data.probabilities.ad,
        predictedClass: data.predicted_class,
      });
      setHeatmapImage(data.heatmap);
      setDemoMode(data.demo_mode);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsScanning(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
    e.target.value = '';
  };

  const getRiskLevel = () => {
    if (!results) return null;
    if (results.ad >= 60) return { label: 'High Risk', color: 'var(--risk-high)' };
    if (results.ad >= 30) return { label: 'Moderate Risk', color: 'var(--risk-med, #f0a500)' };
    return { label: 'Low Risk', color: 'var(--risk-low, #4caf50)' };
  };

  const risk = getRiskLevel();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: '20px' }}>

      {/* Header */}
      <header style={{ marginBottom: '20px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '15px' }}>
        <h1 className="title-primary">ARIA <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>| DIAGNOSTIC DASHBOARD</span></h1>
      </header>

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(250px, 1fr) 2fr minmax(300px, 1fr)', gap: '24px', flex: 1, minHeight: 0 }}>

        {/* Left Column - Patient Info & Upload */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          <div className="panel">
            <span className="subtitle">PATIENT PROFILE</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px' }}>
              <div style={{ width: '50px', height: '50px', borderRadius: '50%', backgroundColor: 'var(--bg-panel-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '20px' }}>👤</div>
              <div>
                <h2 style={{ marginBottom: '4px' }}>John Doe</h2>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>ID: PT-2023051401 | M/68</div>
              </div>
            </div>
            <button style={{ width: '100%' }}>View Full Profile</button>
          </div>

          <div className="panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <span className="subtitle">UPLOAD MEDICAL SCANS</span>

            {/* Hidden real file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />

            <div
              style={{
                border: `2px dashed ${isDragging ? 'var(--primary-teal)' : 'var(--border-subtle)'}`,
                borderRadius: '8px',
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '30px',
                textAlign: 'center',
                cursor: 'pointer',
                backgroundColor: isDragging ? 'rgba(0,200,180,0.05)' : 'var(--bg-dark)',
                transition: 'border-color 0.2s, background-color 0.2s',
                gap: '15px'
              }}
              onClick={handleUpload}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div style={{ fontSize: '40px', color: 'var(--primary-teal)' }}>📁</div>
              {fileName ? (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', wordBreak: 'break-all' }}>
                  {fileName}
                </p>
              ) : (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Drag & drop or click to upload a brain MRI scan (.jpg, .png)
                </p>
              )}
              <button className="primary" onClick={(e) => { e.stopPropagation(); handleUpload(); }}>
                {isScanning ? 'Analyzing...' : 'Browse Files'}
              </button>
            </div>

            {demoMode && (
              <div style={{ marginTop: '10px', padding: '8px', backgroundColor: 'rgba(255,165,0,0.1)', borderRadius: '6px', fontSize: '0.78rem', color: '#f0a500', textAlign: 'center' }}>
                Demo mode — add a trained model to models/ for real predictions
              </div>
            )}

            {error && (
              <div style={{ marginTop: '10px', padding: '8px', backgroundColor: 'rgba(255,71,71,0.1)', borderRadius: '6px', fontSize: '0.78rem', color: 'var(--risk-high)', textAlign: 'center' }}>
                Error: {error}
              </div>
            )}
          </div>

        </div>

        {/* Center Column - Medical Viewer */}
        <div className={`panel ${isScanning ? 'scan-active' : ''}`} style={{ display: 'flex', flexDirection: 'column', padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <span className="subtitle" style={{ margin: 0 }}>AI DIAGNOSTIC ATTENTION HEATMAP</span>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Model: <strong style={{ color: 'var(--text-main)' }}>NeuroResNet v3.1</strong>
            </div>
          </div>

          <div style={{
            flex: 1,
            backgroundColor: '#000',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)',
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden'
          }}>
            {isScanning ? (
              <div style={{ color: 'var(--primary-teal)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '3px solid', borderColor: 'var(--primary-teal) transparent', animation: 'spin 1s linear infinite' }} />
                <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
                Performing Inference...
              </div>
            ) : heatmapImage ? (
              <img
                src={`data:image/png;base64,${heatmapImage}`}
                alt="Grad-CAM Heatmap"
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
            ) : results ? (
              <div style={{ width: '100%', height: '100%', backgroundImage: 'radial-gradient(circle at 50% 40%, rgba(255, 71, 71, 0.4) 0%, transparent 60%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ color: 'var(--text-muted)', textAlign: 'center' }}>
                  <div style={{ fontSize: '80px', marginBottom: '20px' }}>🧠</div>
                  <div style={{ color: 'var(--risk-high)', fontWeight: 'bold' }}>Inference Complete</div>
                  <div style={{ fontSize: '0.85rem', marginTop: '10px' }}>Demo mode — no heatmap available</div>
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)' }}>No scan loaded. Upload a file to begin inference.</div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '10px', marginTop: '15px', justifyContent: 'center' }}>
            <button disabled={!results}>&lt; Prev Slice</button>
            <button disabled={!results}>Next Slice &gt;</button>
          </div>
        </div>

        {/* Right Column - Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          <div className="panel" style={{ opacity: results ? 1 : 0.4, transition: 'opacity 0.3s', flex: 1 }}>
            <span className="subtitle">DIAGNOSTIC RESULTS</span>

            {results ? (
              <h3 style={{ color: risk.color, margin: '15px 0', fontSize: '1.2rem' }}>
                {results.predictedClass.toUpperCase()}: {results.ad}%
              </h3>
            ) : (
              <h3 style={{ color: 'var(--text-muted)', margin: '15px 0', fontSize: '1.2rem' }}>
                AWAITING SCAN...
              </h3>
            )}

            <div style={{ margin: '30px 0' }}>
              <div className="progress-container">
                <div className="progress-header">
                  <span>COGNITIVELY NORMAL</span>
                  <span>{results ? results.cn : 0}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill low" style={{ width: `${results ? results.cn : 0}%` }}></div>
                </div>
              </div>

              <div className="progress-container">
                <div className="progress-header">
                  <span>MILD COGNITIVE IMPAIRMENT</span>
                  <span>{results ? results.mci : 0}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill med" style={{ width: `${results ? results.mci : 0}%` }}></div>
                </div>
              </div>

              <div className="progress-container">
                <div className="progress-header">
                  <span style={{ color: results ? risk.color : 'var(--text-muted)' }}>ALZHEIMER'S DISEASE</span>
                  <span style={{ color: results ? risk.color : 'var(--text-muted)' }}>{results ? results.ad : 0}%</span>
                </div>
                <div className="progress-track" style={{ height: '12px' }}>
                  <div className="progress-fill high" style={{ width: `${results ? results.ad : 0}%` }}></div>
                </div>
              </div>
            </div>
          </div>

          <div className="panel" style={{ opacity: results ? 1 : 0.4, transition: 'opacity 0.3s' }}>
            <span className="subtitle">RISK ASSESSMENT</span>
            <div style={{ display: 'flex', justifyContent: 'space-between', margin: '15px 0 5px', fontSize: '0.9rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Overall Risk</span>
              <strong style={{ color: results ? risk.color : 'var(--text-muted)' }}>
                {results ? risk.label : '--'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', margin: '15px 0 10px', fontSize: '0.9rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Predicted Class</span>
              <strong>{results ? results.predictedClass : '--'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', margin: '15px 0 10px', fontSize: '0.9rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>AD Probability</span>
              <strong style={{ color: results ? risk.color : 'var(--text-muted)' }}>
                {results ? `${results.ad}%` : '--'}
              </strong>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
