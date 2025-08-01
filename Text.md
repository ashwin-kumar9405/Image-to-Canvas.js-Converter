# Overview

This is an AI-powered UI-to-Canvas converter that transforms UI mockups and designs into interactive Canvas.js code. The application uses machine learning (YOLOv8) for UI element detection, OCR (EasyOCR) for text extraction, and generates production-ready Canvas code in multiple frameworks including Konva.js, Fabric.js, and native HTML5 Canvas.

The system processes uploaded images (PNG, JPEG), PDFs, or Figma exports to automatically detect UI components like buttons, text fields, images, and other interface elements, then generates corresponding Canvas.js implementations with accurate positioning, styling, and interactivity.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: React 18 with TypeScript and Vite for fast development and building
- **UI Components**: Shadcn/ui component library built on Radix UI primitives for accessible, customizable components
- **Styling**: Tailwind CSS with CSS variables for theming and responsive design
- **State Management**: TanStack Query (React Query) for server state management and API caching
- **Routing**: Wouter for lightweight client-side routing
- **File Structure**: Organized with clear separation between components, pages, hooks, and utilities

## Backend Architecture
- **Runtime**: Node.js with Express.js server framework
- **Language**: TypeScript with ES modules for type safety and modern JavaScript features
- **Development Setup**: Hot reload with Vite integration for seamless development experience
- **API Design**: RESTful endpoints with structured error handling and request logging
- **File Processing**: Multer middleware for handling multipart file uploads with size and type validation

## Data Storage Solutions
- **Database**: PostgreSQL configured through Drizzle ORM for type-safe database operations
- **Connection**: Neon Database serverless PostgreSQL for scalable cloud database hosting
- **Schema Management**: Drizzle Kit for database migrations and schema evolution
- **Storage Strategy**: In-memory fallback storage implementation for development and testing
- **Data Models**: Structured tables for users, projects, detected elements, and generated code

## Machine Learning Integration
- **UI Detection**: YOLOv8 model integration for automated UI element recognition and classification
- **OCR Processing**: EasyOCR service for text extraction from detected UI components
- **Element Types**: Support for buttons, text fields, images, checkboxes, toggles, and other common UI elements
- **Confidence Scoring**: Machine learning confidence metrics for detection accuracy assessment

## Code Generation Engine
- **Multi-Framework Support**: Generates code for Konva.js, Fabric.js, native Canvas, and React-Canvas
- **Template System**: Modular code generation with framework-specific templates and patterns
- **Optimization**: Code size optimization and accuracy scoring for generated implementations
- **Export Formats**: Support for JavaScript, TypeScript, and React component export formats

## Processing Pipeline
- **File Upload**: Secure file handling with validation for images, PDFs, and Figma files
- **ML Detection**: Automated UI element detection using computer vision models
- **OCR Extraction**: Text content extraction from detected interface components
- **Code Generation**: Framework-specific Canvas code generation with proper event handling
- **Preview System**: Real-time canvas preview with bounding box overlays and element visualization

# External Dependencies

## Core Framework Dependencies
- **@neondatabase/serverless**: Serverless PostgreSQL client for Neon Database connectivity
- **drizzle-orm**: Type-safe ORM with PostgreSQL dialect for database operations
- **express**: Web application framework for Node.js backend API development
- **multer**: Middleware for handling multipart/form-data file uploads

## Frontend UI Libraries
- **@radix-ui/***: Comprehensive set of accessible, unstyled UI primitives for building design systems
- **@tanstack/react-query**: Data fetching and state management library for React applications
- **tailwindcss**: Utility-first CSS framework for rapid UI development
- **class-variance-authority**: Utility for creating variant-based component APIs

## Development Tools
- **vite**: Next-generation frontend build tool with fast hot module replacement
- **tsx**: TypeScript execution environment for Node.js development
- **esbuild**: Fast JavaScript bundler for production builds
- **@replit/vite-plugin-runtime-error-modal**: Development error overlay for Replit environment

## Machine Learning Libraries
- **Computer Vision**: YOLOv8 integration for UI element detection and classification
- **OCR Processing**: EasyOCR library for optical character recognition and text extraction
- **Image Processing**: Canvas and image manipulation utilities for file processing

## Utility Libraries
- **react-hook-form**: Performant forms library with minimal re-renders and validation
- **date-fns**: Modern JavaScript date utility library for date manipulation
- **embla-carousel-react**: Carousel component for image and content sliding
- **cmdk**: Command palette component for enhanced user interactions
