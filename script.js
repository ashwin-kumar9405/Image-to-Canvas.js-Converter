// DOM Elements
const fileUpload = document.getElementById('file-upload');
const fileName = document.getElementById('file-name');
const previewCanvas = document.getElementById('preview-canvas');
const canvasCode = document.getElementById('canvas-code');
const copyButton = document.getElementById('copy-button');
const ctx = previewCanvas.getContext('2d');

// Global variables
let currentImage = null;

// Initialize the application
function init() {
    // Set up event listeners
    fileUpload.addEventListener('change', handleFileUpload);
    copyButton.addEventListener('click', copyCodeToClipboard);
    
    // Set initial canvas size
    resizeCanvas();
    
    // Handle window resize
    window.addEventListener('resize', resizeCanvas);
    
    // Display initial message on canvas
    displayInitialMessage();
}

// Resize canvas based on container size
function resizeCanvas() {
    const container = document.querySelector('.canvas-container');
    previewCanvas.width = container.clientWidth - 40; // Subtract padding
    previewCanvas.height = container.clientHeight - 40;
    
    // Redraw if we have an image
    if (currentImage) {
        drawImageOnCanvas(currentImage);
    } else {
        displayInitialMessage();
    }
}

// Display initial message on canvas
function displayInitialMessage() {
    ctx.fillStyle = '#bdc3c7';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Upload an image or PDF to see preview', previewCanvas.width / 2, previewCanvas.height / 2);
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Update file name display
    fileName.textContent = file.name;
    
    // Process file based on type
    const fileType = file.type.toLowerCase();
    
    try {
        if (fileType.includes('pdf')) {
            await processPDF(file);
        } else if (fileType.includes('image')) {
            await processImage(file);
        } else {
            throw new Error('Unsupported file type. Please upload PNG, JPEG, or PDF.');
        }
    } catch (error) {
        displayError(error.message);
    }
}

// Process image files (PNG, JPEG)
async function processImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                currentImage = img;
                drawImageOnCanvas(img);
                generateCanvasCode(img);
                resolve();
            };
            img.onerror = function() {
                reject(new Error('Failed to load image.'));
            };
            img.src = event.target.result;
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read file.'));
        };
        
        reader.readAsDataURL(file);
    });
}

// Process PDF files
async function processPDF(file) {
    // Load PDF.js if not already loaded
    if (!window.pdfjsLib) {
        window.pdfjsLib = window['pdfjs-dist/build/pdf'];
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.10.377/pdf.worker.min.js';
    }
    
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const page = await pdf.getPage(1); // Get first page
        
        const viewport = page.getViewport({ scale: 1.0 });
        
        // Set canvas dimensions to match PDF page
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        
        // Render PDF page to canvas
        await page.render({
            canvasContext: context,
            viewport: viewport
        }).promise;
        
        // Convert to image and process
        const img = new Image();
        img.onload = function() {
            currentImage = img;
            drawImageOnCanvas(img);
            generateCanvasCode(img);
        };
        img.src = canvas.toDataURL('image/png');
        
    } catch (error) {
        throw new Error('Failed to process PDF: ' + error.message);
    }
}

// Draw image on canvas with proper scaling
function drawImageOnCanvas(img) {
    // Clear canvas
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    // Calculate scaling to fit image within canvas while maintaining aspect ratio
    const scale = Math.min(
        previewCanvas.width / img.width,
        previewCanvas.height / img.height
    );
    
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    
    // Center image on canvas
    const x = (previewCanvas.width - scaledWidth) / 2;
    const y = (previewCanvas.height - scaledHeight) / 2;
    
    // Draw image
    ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
}

// Generate Canvas.js code from image
function generateCanvasCode(img) {
    // Get image data for processing
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // Set to original image dimensions
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    
    // Draw image at original size
    tempCtx.drawImage(img, 0, 0);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Generate Canvas.js code
    let code = `// Canvas.js code for the uploaded image\n`;
    code += `// Original dimensions: ${img.width}x${img.height}\n\n`;
    code += `// Create a canvas element\n`;
    code += `const canvas = document.createElement('canvas');\n`;
    code += `canvas.width = ${img.width};\n`;
    code += `canvas.height = ${img.height};\n`;
    code += `document.body.appendChild(canvas);\n\n`;
    code += `// Get the canvas context\n`;
    code += `const ctx = canvas.getContext('2d');\n\n`;
    
    // Add code to draw the image
    code += `// Draw the image\n`;
    
    // For simplicity, we'll use a data URL approach for the generated code
    // This is more efficient than trying to recreate the image with individual pixel operations
    const dataURL = tempCanvas.toDataURL('image/png');
    
    code += `const img = new Image();\n`;
    code += `img.onload = function() {\n`;
    code += `  ctx.drawImage(img, 0, 0);\n`;
    code += `};\n`;
    code += `img.src = "${dataURL}";\n`;
    
    // Add the code to the textarea
    canvasCode.value = code;
}

// Display error message
function displayError(message) {
    // Clear canvas
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    // Display error message
    ctx.fillStyle = '#e74c3c';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, previewCanvas.width / 2, previewCanvas.height / 2);
    
    // Clear code textarea
    canvasCode.value = '';
}

// Copy code to clipboard
function copyCodeToClipboard() {
    canvasCode.select();
    document.execCommand('copy');
    
    // Visual feedback
    const originalText = copyButton.textContent;
    copyButton.textContent = 'Copied!';
    copyButton.style.backgroundColor = '#27ae60';
    
    // Reset after a short delay
    setTimeout(() => {
        copyButton.textContent = originalText;
        copyButton.style.backgroundColor = '#2ecc71';
    }, 2000);
    
    // Add confetti effect
    confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
    });
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
