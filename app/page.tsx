"use client"

import type React from "react"

import { useState, useCallback, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, Download, Copy, Eye, Code, Zap, CheckCircle, AlertCircle } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface DetectedElement {
  id: string
  type: string
  x: number
  y: number
  width: number
  height: number
  text: string
  confidence: number
}

interface ProcessingResult {
  elements: DetectedElement[]
  accuracy: number
  processingTime: number
  canvasCode: string
  htmlCanvasCode: string
}

export default function ImageToCanvasConverter() {
  const [file, setFile] = useState<File | null>(null)
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<ProcessingResult | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { toast } = useToast()

  const handleFileSelect = useCallback(
    (selectedFile: File) => {
      if (!selectedFile) return

      const validTypes = ["image/png", "image/jpeg", "image/jpg", "application/pdf"]
      if (!validTypes.includes(selectedFile.type)) {
        toast({
          title: "Invalid file type",
          description: "Please upload PNG, JPEG, or PDF files only.",
          variant: "destructive",
        })
        return
      }

      setFile(selectedFile)

      // Create preview URL for images
      if (selectedFile.type.startsWith("image/")) {
        const url = URL.createObjectURL(selectedFile)
        setPreviewUrl(url)
      }
    },
    [toast],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)

      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile) {
        handleFileSelect(droppedFile)
      }
    },
    [handleFileSelect],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const processFile = async () => {
    if (!file) return

    setProcessing(true)
    setProgress(0)
    setResult(null)

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90))
      }, 200)

      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setProgress(100)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Processing failed")
      }

      const data = await response.json()
      setResult(data)

      // Draw detection results on canvas
      if (canvasRef.current && previewUrl) {
        drawDetections(data.elements)
      }

      toast({
        title: "Processing complete!",
        description: `Detected ${data.elements.length} elements with ${data.accuracy}% accuracy.`,
      })
    } catch (error) {
      console.error("Processing error:", error)
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "An unexpected error occurred.",
        variant: "destructive",
      })
    } finally {
      setProcessing(false)
      setTimeout(() => setProgress(0), 1000)
    }
  }

  const drawDetections = (elements: DetectedElement[]) => {
    const canvas = canvasRef.current
    if (!canvas || !previewUrl) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width
      canvas.height = img.height

      // Draw the image
      ctx.drawImage(img, 0, 0)

      // Draw detection boxes
      elements.forEach((element, index) => {
        const color = getElementColor(element.type)

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(element.x, element.y, element.width, element.height)

        // Draw label background
        ctx.fillStyle = color
        ctx.fillRect(element.x, element.y - 20, 100, 20)

        // Draw label text
        ctx.fillStyle = "white"
        ctx.font = "12px Arial"
        ctx.fillText(`${element.type} (${Math.round(element.confidence)}%)`, element.x + 2, element.y - 6)
      })
    }
    img.src = previewUrl
  }

  const getElementColor = (type: string): string => {
    const colors: Record<string, string> = {
      button: "#3b82f6",
      text: "#10b981",
      image: "#f59e0b",
      container: "#8b5cf6",
      input: "#ef4444",
      default: "#6b7280",
    }
    return colors[type] || colors.default
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast({
        title: "Copied!",
        description: "Code copied to clipboard.",
      })
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Please copy the code manually.",
        variant: "destructive",
      })
    }
  }

  const downloadCode = (code: string, filename: string) => {
    const blob = new Blob([code], { type: "text/javascript" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const getAccuracyBadgeVariant = (accuracy: number) => {
    if (accuracy >= 80) return "default"
    if (accuracy >= 60) return "secondary"
    return "destructive"
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">Image to Canvas.js Converter</h1>
          <p className="text-lg text-gray-600">
            Upload images, PDFs, or Figma exports to automatically detect UI elements and generate Canvas.js code
          </p>
        </div>

        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload File
            </CardTitle>
            <CardDescription>Supports PNG, JPEG, and PDF files. Drag and drop or click to select.</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`upload-area ${dragOver ? "dragover" : ""}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.pdf"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                className="hidden"
              />

              {file ? (
                <div className="space-y-2">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto" />
                  <p className="text-lg font-medium">{file.name}</p>
                  <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                  <p className="text-lg font-medium">Drop your file here or click to browse</p>
                  <p className="text-sm text-gray-500">PNG, JPEG, or PDF files only</p>
                </div>
              )}
            </div>

            {file && (
              <div className="mt-4 flex gap-2">
                <Button onClick={processFile} disabled={processing} className="flex-1">
                  {processing ? (
                    <>
                      <Zap className="h-4 w-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Eye className="h-4 w-4 mr-2" />
                      Analyze Image
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setFile(null)
                    setPreviewUrl(null)
                    setResult(null)
                  }}
                >
                  Clear
                </Button>
              </div>
            )}

            {processing && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Processing...</span>
                  <span>{progress}%</span>
                </div>
                <Progress value={progress} className="w-full" />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Preview */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Eye className="h-5 w-5" />
                    Detection Preview
                  </span>
                  <Badge variant={getAccuracyBadgeVariant(result.accuracy)}>{result.accuracy}% Accuracy</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="canvas-preview">
                  <canvas ref={canvasRef} className="max-w-full h-auto" style={{ maxHeight: "400px" }} />
                </div>

                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Elements Detected:</span>
                    <span className="font-medium">{result.elements.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Processing Time:</span>
                    <span className="font-medium">{result.processingTime}ms</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Detected Elements */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5" />
                  Detected Elements
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="element-list">
                  {result.elements.map((element) => (
                    <div key={element.id} className="element-item">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline" className="element-type">
                            {element.type}
                          </Badge>
                          <span className="text-sm font-medium">{element.text}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                          Position: ({element.x}, {element.y}) Size: {element.width}×{element.height}
                        </div>
                        <div className="mt-1">
                          <div className="confidence-bar">
                            <div className="confidence-fill bg-blue-500" style={{ width: `${element.confidence}%` }} />
                          </div>
                          <div className="text-xs text-gray-500 mt-1">Confidence: {element.confidence}%</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Generated Code */}
        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                Generated Code
              </CardTitle>
              <CardDescription>Choose between Canvas.js charts or HTML5 Canvas implementation</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="canvasjs" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="canvasjs">Canvas.js Chart</TabsTrigger>
                  <TabsTrigger value="html5">HTML5 Canvas</TabsTrigger>
                </TabsList>

                <TabsContent value="canvasjs" className="space-y-4">
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={() => copyToClipboard(result.canvasCode)}>
                      <Copy className="h-4 w-4 mr-2" />
                      Copy
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => downloadCode(result.canvasCode, "canvas-chart.js")}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                  <Textarea
                    value={result.canvasCode}
                    readOnly
                    className="code-output min-h-[300px] font-mono text-sm"
                  />
                </TabsContent>

                <TabsContent value="html5" className="space-y-4">
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={() => copyToClipboard(result.htmlCanvasCode)}>
                      <Copy className="h-4 w-4 mr-2" />
                      Copy
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => downloadCode(result.htmlCanvasCode, "html5-canvas.js")}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                  <Textarea
                    value={result.htmlCanvasCode}
                    readOnly
                    className="code-output min-h-[300px] font-mono text-sm"
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}

        {/* Info Alert */}
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            This tool uses advanced computer vision to detect UI elements in your images. For best results, use
            high-quality screenshots with clear element boundaries. The accuracy score reflects the confidence in
            detected elements and text extraction.
          </AlertDescription>
        </Alert>
      </div>
    </div>
  )
}
