"""
Enhanced FastAPI Backend for Image-to-Canvas.js Converter
Integrates all ML processing modules with improved error handling and performance
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import shutil
import time
import asyncio
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Import our custom modules
from ml_processor import LayoutDetector, CanvasJSGenerator, calculate_accuracy_score
from pdf_processor import PDFProcessor
from figma_processor import FigmaProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="Image-to-Canvas.js Converter API",
    description="Advanced ML-powered converter for images, PDFs, and Figma files to Canvas.js code",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize processors
layout_detector = LayoutDetector()
canvas_generator = CanvasJSGenerator()
pdf_processor = PDFProcessor()
figma_processor = FigmaProcessor()

# Global processing cache
processing_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Image-to-Canvas.js Converter API")
    logger.info("Loading ML models...")
    
    # Warm up the models
    try:
        # Create a small test image to warm up YOLO
        import cv2
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("temp/warmup.jpg", test_img)
        
        # Warm up detection
        layout_detector.detect_elements("temp/warmup.jpg")
        
        # Clean up
        os.remove("temp/warmup.jpg")
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Image-to-Canvas.js Converter API",
        "version": "2.0.0",
        "endpoints": {
            "process": "/api/process - Process image/PDF/Figma files",
            "status": "/api/status/{task_id} - Check processing status",
            "health": "/api/health - Health check"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": True
    }

@app.post("/api/process")
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process uploaded file and extract layout elements"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (max 50MB)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    # Check file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.fig', '.json'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique task ID
    task_id = f"task_{int(time.time() * 1000)}"
    
    # Save uploaded file
    temp_file_path = f"temp/{task_id}_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file based on type
        start_time = time.time()
        
        if file_extension in ['.png', '.jpg', '.jpeg']:
            result = await process_image_file(temp_file_path, task_id)
        elif file_extension == '.pdf':
            result = await process_pdf_file(temp_file_path, task_id)
        elif file_extension in ['.fig', '.json']:
            result = await process_figma_file(temp_file_path, task_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['task_id'] = task_id
        
        # Cache result for later retrieval
        processing_cache[task_id] = result
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, temp_file_path, task_id)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Processing error for {file.filename}: {e}")
        
        # Cleanup on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def process_image_file(file_path: str, task_id: str) -> Dict:
    """Process image file using ML detection"""
    try:
        # Run detection
        detections = layout_detector.detect_elements(file_path)
        
        # Calculate accuracy score
        accuracy_score = calculate_accuracy_score(detections)
        
        # Generate Canvas.js code
        canvas_code = canvas_generator.generate_chart_code(detections)
        html5_code = canvas_generator.generate_html5_canvas_code(detections)
        
        # Create preview URL (base64 encoded)
        import base64
        with open(file_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            preview_url = f"data:image/jpeg;base64,{img_data}"
        
        return {
            "layout": detections,
            "canvas_js_code": canvas_code,
            "html5_canvas_code": html5_code,
            "preview_url": preview_url,
            "accuracy_score": accuracy_score,
            "element_count": len(detections),
            "text_elements": len([d for d in detections if d['text']]),
            "processing_type": "image_ml"
        }
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise

async def process_pdf_file(file_path: str, task_id: str) -> Dict:
    """Process PDF file"""
    try:
        # Process first page of PDF
        detections, image = pdf_processor.process_pdf(file_path, 0)
        
        # Calculate accuracy score
        accuracy_score = calculate_accuracy_score(detections)
        
        # Generate Canvas.js code
        canvas_code = canvas_generator.generate_chart_code(detections)
        html5_code = canvas_generator.generate_html5_canvas_code(detections)
        
        # Save processed image and create preview URL
        import cv2
        import base64
        preview_path = f"temp/{task_id}_preview.jpg"
        cv2.imwrite(preview_path, image)
        
        with open(preview_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            preview_url = f"data:image/jpeg;base64,{img_data}"
        
        return {
            "layout": detections,
            "canvas_js_code": canvas_code,
            "html5_canvas_code": html5_code,
            "preview_url": preview_url,
            "accuracy_score": accuracy_score,
            "element_count": len(detections),
            "text_elements": len([d for d in detections if d['text']]),
            "processing_type": "pdf"
        }
        
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise

async def process_figma_file(file_path: str, task_id: str) -> Dict:
    """Process Figma JSON file"""
    try:
        # Load JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            figma_data = json.load(f)
        
        # Process Figma data
        detections = figma_processor.process_figma_file(figma_data)
        
        # Calculate accuracy score
        accuracy_score = calculate_accuracy_score(detections)
        
        # Generate Canvas.js code
        canvas_code = canvas_generator.generate_chart_code(detections)
        html5_code = canvas_generator.generate_html5_canvas_code(detections)
        
        return {
            "layout": detections,
            "canvas_js_code": canvas_code,
            "html5_canvas_code": html5_code,
            "preview_url": "",  # No preview for Figma JSON
            "accuracy_score": accuracy_score,
            "element_count": len(detections),
            "text_elements": len([d for d in detections if d['text']]),
            "processing_type": "figma"
        }
        
    except Exception as e:
        logger.error(f"Figma processing error: {e}")
        raise

@app.get("/api/status/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing status for a task"""
    if task_id in processing_cache:
        return JSONResponse(content={
            "status": "completed",
            "result": processing_cache[task_id]
        })
    else:
        return JSONResponse(content={
            "status": "not_found",
            "message": "Task not found or expired"
        })

@app.post("/api/batch-process")
async def batch_process_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Process multiple files in batch"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    batch_id = f"batch_{int(time.time() * 1000)}"
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file
            task_id = f"{batch_id}_file_{i}"
            
            # Save file
            temp_file_path = f"temp/{task_id}_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Determine file type and process
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension in ['.png', '.jpg', '.jpeg']:
                result = await process_image_file(temp_file_path, task_id)
            elif file_extension == '.pdf':
                result = await process_pdf_file(temp_file_path, task_id)
            elif file_extension in ['.fig', '.json']:
                result = await process_figma_file(temp_file_path, task_id)
            else:
                result = {"error": f"Unsupported file type: {file_extension}"}
            
            result['filename'] = file.filename
            result['task_id'] = task_id
            results.append(result)
            
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_files, temp_file_path, task_id)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "batch_id": batch_id,
        "results": results,
        "total_files": len(files),
        "successful": len([r for r in results if "error" not in r])
    })

async def cleanup_temp_files(file_path: str, task_id: str):
    """Clean up temporary files after processing"""
    await asyncio.sleep(3600)  # Wait 1 hour before cleanup
    
    try:
        # Remove temp file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove preview file if exists
        preview_path = f"temp/{task_id}_preview.jpg"
        if os.path.exists(preview_path):
            os.remove(preview_path)
        
        # Remove from cache
        if task_id in processing_cache:
            del processing_cache[task_id]
            
        logger.info(f"Cleaned up files for task {task_id}")
        
    except Exception as e:
        logger.warning(f"Cleanup failed for {task_id}: {e}")

@app.get("/api/stats")
async def get_processing_stats():
    """Get processing statistics"""
    return {
        "active_tasks": len(processing_cache),
        "temp_files": len(os.listdir("temp")) if os.path.exists("temp") else 0,
        "supported_formats": [".png", ".jpg", ".jpeg", ".pdf", ".fig", ".json"],
        "max_file_size": "50MB",
        "max_batch_size": 10
    }

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
