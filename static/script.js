// ============================================================================
// MobForge-Net Web Interface JavaScript
// ============================================================================

// ────────────────────────────────────────
// DOM Elements
// ────────────────────────────────────────

const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resetBtn = document.getElementById('resetBtn');
const downloadBtn = document.getElementById('downloadBtn');
const statusText = document.getElementById('statusText');
const modelStatus = document.querySelector('.model-status');

// Result display elements
const visualizationImg = document.getElementById('visualizationImg');
const verdictText = document.getElementById('verdictText');
const forgeryPctText = document.getElementById('forgeryPctText');
const inferenceTimeText = document.getElementById('inferenceTimeText');
const deviceText = document.getElementById('deviceText');

// ────────────────────────────────────────
// State
// ────────────────────────────────────────

let currentVisualization = null;
let modelReady = false;

// ────────────────────────────────────────
// Event Listeners
// ────────────────────────────────────────

// Upload box click
uploadBox.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

// Reset button
resetBtn.addEventListener('click', resetUI);

// Download button
downloadBtn.addEventListener('click', downloadResult);

// ────────────────────────────────────────
// Functions
// ────────────────────────────────────────

/**
 * Check model status on page load
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'ready') {
            statusText.textContent = 'Ready';
            modelReady = true;
            modelStatus.style.opacity = '1';
        } else {
            statusText.textContent = 'Error';
            modelStatus.querySelector('.status-dot').style.background = '#dc2626';
            modelReady = false;
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        statusText.textContent = 'Offline';
        modelStatus.querySelector('.status-dot').style.background = '#f59e0b';
        modelReady = false;
    }
}

/**
 * Handle file selection and upload
 */
async function handleFileSelect(file) {
    // Validate file
    const validExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff'];
    const fileExt = file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExt)) {
        showError(`Invalid file type: ${fileExt}. Supported: ${validExtensions.join(', ')}`);
        return;
    }
    
    if (file.size > 50 * 1024 * 1024) {
        showError('File too large. Maximum size: 50MB');
        return;
    }
    
    // Show loading
    hideError();
    hideResults();
    loadingSpinner.style.display = 'flex';
    uploadBox.style.opacity = '0.5';
    uploadBox.style.pointerEvents = 'none';
    
    try {
        // Upload file
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Upload failed');
        }
        
        // Display results
        displayResults(data);
        currentVisualization = data.visualization;
        
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        loadingSpinner.style.display = 'none';
        uploadBox.style.opacity = '1';
        uploadBox.style.pointerEvents = 'auto';
    }
}

/**
 * Display inference results
 */
function displayResults(data) {
    // Set visualization
    visualizationImg.src = data.visualization;
    
    // Set metrics
    verdictText.textContent = data.verdict;
    verdictText.className = 'metric-value verdict ' + (data.verdict === 'FORGED' ? 'forged' : 'authentic');
    
    forgeryPctText.textContent = `${data.forgery_pct}%`;
    inferenceTimeText.textContent = `${data.inference_ms}ms`;
    deviceText.textContent = data.device === 'cpu' ? 'CPU' : 'GPU';
    
    // Show results section
    resultsSection.style.display = 'grid';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Reset UI to initial state
 */
function resetUI() {
    fileInput.value = '';
    hideResults();
    hideError();
    uploadBox.style.opacity = '1';
    uploadBox.style.pointerEvents = 'auto';
    uploadBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Download visualization result
 */
function downloadResult() {
    if (!currentVisualization) {
        showError('No result to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = currentVisualization;
    link.download = `forgery-detection-${Date.now()}.png`;
    link.click();
}

/**
 * Show error message
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

// ────────────────────────────────────────
// Initialization
// ────────────────────────────────────────

// Check model status on page load
window.addEventListener('load', () => {
    checkModelStatus();
    // Recheck every 30 seconds
    setInterval(checkModelStatus, 30000);
});

console.log('🔍 MobForge-Net Web Interface Ready');
