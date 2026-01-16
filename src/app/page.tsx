"use client";

import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "/wasm/";

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("준비 중...");
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [labels, setLabels] = useState<string[]>([]);

  const [image, setImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    async function init() {
      setMounted(true);
      try {
        setLoadingStatus("모델 및 레이블 로딩 중...");
        
        // 레이블 로드
        const labelsRes = await fetch("/model/labels.txt");
        const labelsText = await labelsRes.text();
        const labelsArray = labelsText.split("\n").map(line => line.trim()).filter(line => line !== "");
        setLabels(labelsArray);

        // 모델 세션 생성
        const sess = await ort.InferenceSession.create("/model/mobilenetv2.onnx", {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        setSession(sess);
        setIsModelLoaded(true);
        setLoadingStatus("모델 준비 완료!");
      } catch (error) {
        console.error("로딩 실패:", error);
        setLoadingStatus("로딩에 실패했습니다.");
      }
    }
    init();
  }, []);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const objUrl = URL.createObjectURL(file);
      setImage(objUrl);
      setResult(null);
    }
  };

  const runInference = async () => {
    if (!session || !image || labels.length === 0) return;
    setIsProcessing(true);
    setResult("분석 중...");
    
    try {
      // 1. 이미지 객체 생성 및 로드 대기
      const img = new window.Image();
      img.src = image;
      await new Promise((resolve) => (img.onload = resolve)); 

      // 2. 캔버스를 이용한 리사이징
      const canvas = document.createElement("canvas");
      canvas.width = 224; canvas.height = 224;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas context is null");
      
      ctx.drawImage(img, 0, 0, 224, 224);
      const imageData = ctx.getImageData(0, 0, 224, 224).data;

      // 3. 정규화 (MobileNetV2 기준)
      const float32Data = new Float32Array(3 * 224 * 224);
      for (let i = 0; i < 224 * 224; i++) {
        float32Data[i] = (imageData[i * 4] / 255.0 - 0.485) / 0.229;
        float32Data[i + 50176] = (imageData[i * 4 + 1] / 255.0 - 0.456) / 0.224;
        float32Data[i + 100352] = (imageData[i * 4 + 2] / 255.0 - 0.406) / 0.225;
      }

      // 4. 추론 실행
      const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
      const outputData = await session.run({ [session.inputNames[0]]: inputTensor });
      const output = outputData[session.outputNames[0]].data as Float32Array;

      // 5. 최댓값 찾기
      let maxIdx = 0, maxVal = -Infinity;
      for (let i = 0; i < output.length; i++) {
        if (output[i] > maxVal) {
          maxVal = output[i];
          maxIdx = i;
        }
      }
      
      const className = labels[maxIdx] || `Unknown (${maxIdx})`;
      setResult(`결과: ${className} (신뢰도: ${maxVal.toFixed(2)})`);
    } catch (error) {
      console.error(error);
      setResult("에러 발생");
    } finally {
      setIsProcessing(false);
    }
  };

  if (!mounted) return null;

  return (
    <main style={{ width: "100%", maxWidth: "500px", padding: "20px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "20px" }}>
      <header>
        <h1 style={{ fontSize: "24px", fontWeight: "800", margin: "0 0 8px 0" }}>Vision AI</h1>
        <p style={{ fontSize: "14px", color: "#666", margin: 0 }}>{loadingStatus}</p>
      </header>

      {isModelLoaded && session && (
        <div style={{ padding: "12px", backgroundColor: "#f0f7ff", color: "#0070f3", borderRadius: "10px", fontSize: "12px", fontWeight: "600" }}>
          ✅ 모델 준비 완료
        </div>
      )}

      <div 
        onClick={() => fileInputRef.current?.click()}
        style={{ width: "100%", height: "250px", border: "2px dashed #e5e7eb", borderRadius: "20px", position: "relative", cursor: "pointer", overflow: "hidden", backgroundColor: "#f9fafb" }}
      >
        <input type="file" ref={fileInputRef} onChange={handleImageUpload} accept="image/*" style={{ display: "none" }} />
        {image ? (
          <Image src={image} alt="Uploaded" fill style={{ objectFit: "cover" }} unoptimized />
        ) : (
          <div style={{ textAlign: "center", color: "#9ca3af", paddingTop: "90px" }}>
            <span style={{ fontSize: "40px" }}>📸</span>
            <p>이미지 업로드</p>
          </div>
        )}
      </div>

      {image && isModelLoaded && (
        <button
          onClick={runInference}
          disabled={isProcessing}
          style={{ width: "100%", padding: "16px", borderRadius: "14px", backgroundColor: "#0070f3", color: "white", border: "none", fontWeight: "600", cursor: "pointer" }}
        >
          {isProcessing ? "분석 중..." : "이미지 분석하기"}
        </button>
      )}

      {result && (
        <div style={{ padding: "20px", backgroundColor: "#f0fdf4", borderRadius: "16px", border: "1px solid #dcfce7" }}>
          <p style={{ fontSize: "16px", color: "#14532d", fontWeight: "600", margin: 0 }}>{result}</p>
        </div>
      )}
    </main>
  );
}
