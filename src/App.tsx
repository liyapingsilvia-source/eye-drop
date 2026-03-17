/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import Matter from 'matter-js';
import { Send, Camera, Eye, Trash2, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export default function App() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Matter.Engine | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const handCollidersRef = useRef<Map<string, Matter.Body>>(new Map());
  const requestRef = useRef<number | null>(null);
  const lastEyePosRef = useRef<{ left: { x: number, y: number }, right: { x: number, y: number } } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // State
  const [inputText, setInputText] = useState('');
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isFaceModelLoaded, setIsFaceModelLoaded] = useState(false);
  const [isHandModelLoaded, setIsHandModelLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(true);
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });
  const [fontSize, setFontSize] = useState(32);
  const fontSizeRef = useRef(32);
  const prevFontSizeRef = useRef(32);
  const [fallSpeed, setFallSpeed] = useState(1);
  const fallSpeedRef = useRef(1);

  // Sync refs with state
  useEffect(() => {
    fontSizeRef.current = fontSize;
  }, [fontSize]);

  useEffect(() => {
    fallSpeedRef.current = fallSpeed;
  }, [fallSpeed]);

  // Sync physics gravity with fallSpeed
  useEffect(() => {
    if (!engineRef.current) return;
    engineRef.current.world.gravity.y = fallSpeed;
  }, [fallSpeed]);

  // Sync existing bodies with font size changes
  useEffect(() => {
    if (!engineRef.current) return;
    const world = engineRef.current.world;
    const bodies = Matter.Composite.allBodies(world);
    const scaleFactor = fontSize / prevFontSizeRef.current;
    
    bodies.forEach(body => {
      if (!body.isStatic) {
        Matter.Body.scale(body, scaleFactor, scaleFactor);
      }
    });
    
    prevFontSizeRef.current = fontSize;
  }, [fontSize]);

  // Suppress the confusing "INFO: Created TensorFlow Lite XNNPACK delegate for CPU" log
  useEffect(() => {
    const originalInfo = console.info;
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;
    
    const filter = (args: any[], ori: any) => {
      const msg = args[0];
      if (msg && typeof msg === 'string' && (
        msg.includes('XNNPACK') || 
        msg.includes('TensorFlow Lite') || 
        msg.includes('delegate') ||
        msg.includes('Created TensorFlow')
      )) {
        return;
      }
      ori.apply(console, args);
    };

    console.info = (...args) => filter(args, originalInfo);
    console.log = (...args) => filter(args, originalLog);
    console.warn = (...args) => filter(args, originalWarn);
    console.error = (...args) => filter(args, originalError);

    return () => {
      console.info = originalInfo;
      console.log = originalLog;
      console.warn = originalWarn;
      console.error = originalError;
    };
  }, []);

  // Initialize Physics Engine
  useEffect(() => {
    const engine = Matter.Engine.create();
    const world = engine.world;

    // Create initial boundaries
    const ground = Matter.Bodies.rectangle(dimensions.width / 2, dimensions.height + 50, dimensions.width * 2, 100, { isStatic: true, label: 'ground' });
    const leftWall = Matter.Bodies.rectangle(-50, dimensions.height / 2, 100, dimensions.height * 2, { isStatic: true, label: 'leftWall' });
    const rightWall = Matter.Bodies.rectangle(dimensions.width + 50, dimensions.height / 2, 100, dimensions.height * 2, { isStatic: true, label: 'rightWall' });

    Matter.Composite.add(world, [ground, leftWall, rightWall]);
    engineRef.current = engine;

    return () => {
      Matter.Engine.clear(engine);
    };
  }, []);

  // Handle Resize
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });

        if (engineRef.current) {
          const world = engineRef.current.world;
          const bodies = Matter.Composite.allBodies(world);
          
          // Update boundary positions
          bodies.forEach(body => {
            if (body.label === 'ground') {
              Matter.Body.setPosition(body, { x: width / 2, y: height + 50 });
            } else if (body.label === 'leftWall') {
              Matter.Body.setPosition(body, { x: -50, y: height / 2 });
            } else if (body.label === 'rightWall') {
              Matter.Body.setPosition(body, { x: width + 50, y: height / 2 });
            }
          });
        }
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Initialize Face Landmarker
  useEffect(() => {
    let isMounted = true;
    async function setupFaceLandmarker() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU" 
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });
        
        if (isMounted) {
          faceLandmarkerRef.current = faceLandmarker;
          setIsFaceModelLoaded(true);
        }
      } catch (err) {
        console.error("Failed to load Face Landmarker:", err);
        if (isMounted) {
          setError("Failed to load face tracking model. Please check your internet connection and browser support for WebGL.");
        }
      }
    }
    setupFaceLandmarker();
    return () => {
      isMounted = false;
    };
  }, []);

  // Initialize Hand Landmarker
  useEffect(() => {
    let isMounted = true;
    async function setupHandLandmarker() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        
        if (isMounted) {
          handLandmarkerRef.current = handLandmarker;
          setIsHandModelLoaded(true);
        }
      } catch (err) {
        console.error("Failed to load Hand Landmarker:", err);
      }
    }
    setupHandLandmarker();
    return () => {
      isMounted = false;
    };
  }, []);

  // Setup Camera
  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsCameraReady(true);
          };
        }
      } catch (err) {
        console.error("Camera access denied:", err);
        setError("Camera access denied. Please click the camera icon in your browser's address bar to allow access, then click Retry.");
      }
    }
    setupCamera();
  }, []);

  // Main Loop
  const animate = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !faceLandmarkerRef.current || !engineRef.current) {
      requestRef.current = requestAnimationFrame(animate);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const engine = engineRef.current;

    if (!ctx) return;

    // 1. Update Physics
    if (engineRef.current) {
      engineRef.current.gravity.scale = 0.001 * fallSpeedRef.current;
    }
    Matter.Engine.update(engine, 1000 / 60);

    // 2. Detect Faces & Hands
    let faceResults: any = null;
    let handResults: any = null;
    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      try {
        const startTimeMs = performance.now();
        faceResults = faceLandmarkerRef.current.detectForVideo(video, startTimeMs);
        if (handLandmarkerRef.current) {
          handResults = handLandmarkerRef.current.detectForVideo(video, startTimeMs);
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    // 3. Draw
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const videoAspect = video.videoWidth / video.videoHeight;
    const canvasAspect = dimensions.width / dimensions.height;
    
    let scaleX: number, scaleY: number, offsetX = 0, offsetY = 0;
    
    if (canvasAspect > videoAspect) {
      scaleX = dimensions.width;
      scaleY = dimensions.width / videoAspect;
      offsetY = (dimensions.height - scaleY) / 2;
    } else {
      scaleY = dimensions.height;
      scaleX = dimensions.height * videoAspect;
      offsetX = (dimensions.width - scaleX) / 2;
    }

    const mapPos = (landmark: any) => ({
      x: landmark.x * scaleX + offsetX,
      y: landmark.y * scaleY + offsetY
    });

    // Track eyes for spawning
    if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
      const landmarks = faceResults.faceLandmarks[0];
      const leftEye = landmarks[468];
      const rightEye = landmarks[473];

      if (leftEye && rightEye) {
        const leftPos = mapPos(leftEye);
        const rightPos = mapPos(rightEye);
        lastEyePosRef.current = { left: leftPos, right: rightPos };

        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.arc(leftPos.x, leftPos.y, 4, 0, Math.PI * 2);
        ctx.arc(rightPos.x, rightPos.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Handle Hand Colliders
    const activeHandIds = new Set<string>();
    if (handResults && handResults.landmarks) {
      handResults.landmarks.forEach((hand: any[], handIdx: number) => {
        // We'll use specific landmarks for colliders: fingertips and palm center
        // Fingertip indices: 4, 8, 12, 16, 20
        // Palm center: 0, 9
        const colliderIndices = [0, 4, 8, 12, 16, 20, 9];
        
        colliderIndices.forEach((idx) => {
          const landmark = hand[idx];
          const pos = mapPos(landmark);
          const id = `hand-${handIdx}-landmark-${idx}`;
          activeHandIds.add(id);

          let body = handCollidersRef.current.get(id);
          if (!body) {
            body = Matter.Bodies.circle(pos.x, pos.y, 20, {
              isStatic: true,
              render: { visible: false }
            });
            Matter.Composite.add(engine.world, body);
            handCollidersRef.current.set(id, body);
          } else {
            Matter.Body.setPosition(body, pos);
          }

          // Draw hand feedback
          ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, 15, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.stroke();
        });
      });
    }

    // Remove inactive hand colliders
    handCollidersRef.current.forEach((body, id) => {
      if (!activeHandIds.has(id)) {
        Matter.Composite.remove(engine.world, body);
        handCollidersRef.current.delete(id);
      }
    });

    // 4. Render Physics Bodies
    const bodies = Matter.Composite.allBodies(engine.world);
    const currentFontSize = fontSizeRef.current;
    const fontStack = 'bold ' + currentFontSize + 'px "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Inter", sans-serif';
    
    ctx.font = fontStack;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    bodies.forEach(body => {
      if (body.isStatic) return;

      const { x, y } = body.position;
      const angle = body.angle;
      const text = (body as any).textValue || '';

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);
      
      ctx.font = fontStack;
      
      // Draw text with shadow for readability
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 6;
      ctx.fillStyle = '#ffffff';
      ctx.fillText(text, 0, 0);
      
      ctx.restore();

      if (y > dimensions.height + 100) {
        Matter.Composite.remove(engine.world, body);
      }
    });

    requestRef.current = requestAnimationFrame(animate);
  }, [dimensions]);

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [animate]);

  const spawnText = (text: string) => {
    if (!engineRef.current || !text.trim()) return;

    const chars = text.split('');
    const currentFontSize = fontSizeRef.current;
    const currentFallSpeed = fallSpeedRef.current;
    const bodySize = currentFontSize * 0.8;
    const spawnDelay = 150 / currentFallSpeed; // Adjust delay based on speed

    chars.forEach((char, index) => {
      setTimeout(() => {
        if (!engineRef.current || !lastEyePosRef.current) return;
        
        const { left, right } = lastEyePosRef.current;
        const latestFontSize = fontSizeRef.current;
        const latestBodySize = latestFontSize * 0.8;
        
        [left, right].forEach(pos => {
          const body = Matter.Bodies.rectangle(
            pos.x, 
            pos.y + 20, // Spawn slightly below eye center
            latestBodySize, latestBodySize, 
            {
              restitution: 0.5,
              friction: 0.1,
              angle: (Math.random() - 0.5) * 0.2,
              chamfer: { radius: latestBodySize * 0.2 }
            }
          );
          (body as any).textValue = char;
          Matter.Composite.add(engineRef.current!.world, body);
        });
      }, index * spawnDelay);
    });

    setInputText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      spawnText(inputText);
    }
  };

  const clearAll = () => {
    if (engineRef.current) {
      const bodies = Matter.Composite.allBodies(engineRef.current.world);
      bodies.forEach(body => {
        if (!body.isStatic) {
          Matter.Composite.remove(engineRef.current!.world, body);
        }
      });
    }
  };

  return (
    <div ref={containerRef} className="relative w-screen h-screen bg-black overflow-hidden font-sans">
      {/* Full-screen Video Feed */}
      <video
        ref={videoRef}
        className="absolute inset-0 w-full h-full object-cover"
        playsInline
        muted
      />
      
      {/* Full-screen Physics Canvas */}
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className="absolute inset-0 w-full h-full pointer-events-none z-10"
      />

      {/* Header Overlay */}
      <div className="absolute top-6 left-0 right-0 z-20 pointer-events-none flex flex-col items-center">
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-2xl font-bold tracking-tighter bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent flex items-center justify-center gap-2">
            <Eye className="w-6 h-6 text-emerald-400" />
            Eye Drop Text
          </h1>
          <p className="text-slate-400 mt-1 text-[10px] font-medium italic">Type and watch it fall</p>
        </motion.div>
      </div>

      {/* Top Controls Container */}
      <div className="absolute top-8 left-0 right-0 z-40 flex justify-between px-8 pointer-events-none">
        {/* Left Sliders */}
        <div className="flex flex-col gap-4 pointer-events-auto">
          {/* Font Size Slider */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-44 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[1.5rem] p-4 flex flex-col gap-2.5 shadow-[0_8px_32px_0_rgba(0,0,0,0.3)]"
          >
            <div className="flex justify-between items-center px-1">
              <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-white/50">Size</span>
              <span className="text-[10px] font-mono font-bold text-white">{fontSize}</span>
            </div>
            <input 
              type="range" 
              min="2" 
              max="80" 
              value={fontSize} 
              onChange={(e) => setFontSize(parseInt(e.target.value))}
              className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-white"
            />
          </motion.div>

          {/* Fall Speed Slider */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="w-44 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[1.5rem] p-4 flex flex-col gap-2.5 shadow-[0_8px_32px_0_rgba(0,0,0,0.3)]"
          >
            <div className="flex justify-between items-center px-1">
              <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-white/50">Speed</span>
              <span className="text-[10px] font-mono font-bold text-white">{fallSpeed.toFixed(1)}x</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="5" 
              step="0.1"
              value={fallSpeed} 
              onChange={(e) => setFallSpeed(parseFloat(e.target.value))}
              className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-white"
            />
          </motion.div>
        </div>

        {/* Center Input Box */}
        <div className="flex-1 flex justify-center px-4 pointer-events-auto">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-lg relative group"
          >
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type something..."
              className="w-full bg-white/15 backdrop-blur-3xl border border-white/30 rounded-[2.5rem] py-5 px-8 pr-24 text-white placeholder-white/40 focus:outline-none focus:ring-4 focus:ring-white/10 shadow-[0_20px_50px_rgba(0,0,0,0.3)] transition-all text-xl font-medium"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
              <button 
                onClick={clearAll}
                className="p-3 text-white/40 hover:text-white/80 transition-colors"
                title="Clear all"
              >
                <Trash2 className="w-5 h-5" />
              </button>
              <button
                onClick={() => spawnText(inputText)}
                disabled={!inputText.trim()}
                className="p-4 bg-white text-black rounded-full hover:bg-white/90 disabled:opacity-20 disabled:scale-95 transition-all shadow-xl active:scale-90"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        </div>

        {/* Right Info Box (Placeholder for spacing / Toggle) */}
        <div className="w-44 flex justify-end pointer-events-auto">
          {!showInfo && (
            <button 
              onClick={() => setShowInfo(true)}
              className="w-12 h-12 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-full flex items-center justify-center text-white/70 hover:text-white transition-all shadow-xl"
            >
              <Info className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Status Indicators (Bottom Left) */}
      <div className="absolute bottom-8 left-8 z-30 flex flex-col gap-2 text-[9px] font-bold uppercase tracking-[0.2em] text-white/40">
        <div className="flex items-center gap-3">
          <div className={`w-1.5 h-1.5 rounded-full ${isFaceModelLoaded ? 'bg-white shadow-[0_0_8px_white]' : 'bg-white/20'}`} />
          Face Tracking {isFaceModelLoaded ? 'Online' : 'Offline'}
        </div>
        <div className="flex items-center gap-3">
          <div className={`w-1.5 h-1.5 rounded-full ${isHandModelLoaded ? 'bg-white shadow-[0_0_8px_white]' : 'bg-white/20'}`} />
          Hand Tracking {isHandModelLoaded ? 'Online' : 'Offline'}
        </div>
        <div className="flex items-center gap-3">
          <div className={`w-1.5 h-1.5 rounded-full ${isCameraReady ? 'bg-white shadow-[0_0_8px_white]' : 'bg-white/20'}`} />
          Camera {isCameraReady ? 'Ready' : 'Waiting'}
        </div>
      </div>

      {/* Loading & Error Overlays */}
      <AnimatePresence>
        {(!isCameraReady || !isFaceModelLoaded || !isHandModelLoaded) && !error && (
          <motion.div 
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 backdrop-blur-3xl z-50"
          >
            <div className="w-12 h-12 border-2 border-white/20 border-t-white rounded-full animate-spin mb-8" />
            <p className="text-white font-medium text-xs tracking-[0.3em] uppercase opacity-60">
              Initializing { !isCameraReady ? 'Vision' : 'Core' }
            </p>
          </motion.div>
        )}
        
        {error && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-3xl z-50 p-8 text-center"
          >
            <div className="bg-white/10 p-10 rounded-[3rem] border border-white/20 backdrop-blur-2xl max-w-md shadow-2xl">
              <Camera className="w-12 h-12 text-white mb-8 mx-auto opacity-50" />
              <h2 className="text-white text-2xl font-bold mb-4 tracking-tight">Camera Required</h2>
              <p className="text-white/60 text-sm mb-10 leading-relaxed px-4">
                {error}
                <br /><br />
                <span className="text-[10px] uppercase tracking-widest opacity-50">Check your browser settings</span>
              </p>
              <button 
                onClick={() => window.location.reload()}
                className="w-full py-5 bg-white text-black rounded-[1.5rem] transition-all text-lg font-bold shadow-2xl active:scale-95 hover:bg-white/90"
              >
                Retry Connection
              </button>
            </div>
          </motion.div>
        )}

        {showInfo && isFaceModelLoaded && isHandModelLoaded && isCameraReady && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9, x: 20 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            exit={{ opacity: 0, scale: 0.9, x: 20 }}
            className="absolute top-28 right-8 bg-white/10 backdrop-blur-2xl p-8 rounded-[2.5rem] border border-white/20 max-w-[280px] shadow-[0_30px_60px_rgba(0,0,0,0.4)] z-40"
          >
            <button 
              onClick={() => setShowInfo(false)}
              className="absolute top-6 right-6 text-white/30 hover:text-white transition-colors"
            >
              ×
            </button>
            <div className="flex items-center gap-3 mb-6 text-white/80">
              <Info className="w-5 h-5 opacity-50" />
              <span className="text-[10px] font-bold uppercase tracking-[0.2em]">Guide</span>
            </div>
            <ul className="text-[12px] text-white/70 space-y-4 leading-relaxed">
              <li className="flex gap-4">
                <span className="text-white/20 font-black italic text-lg leading-none">01</span>
                Face the camera clearly to start tracking.
              </li>
              <li className="flex gap-4">
                <span className="text-white/20 font-black italic text-lg leading-none">02</span>
                Type your message in the top bar.
              </li>
              <li className="flex gap-4">
                <span className="text-white/20 font-black italic text-lg leading-none">03</span>
                Press Enter to see words fall from your eyes.
              </li>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
