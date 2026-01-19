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

  // 추가 state
  // 탭 관리
  const [activeTab, setActiveTab] = useState<"single" | "batch">("single");
  // 배치 모드용 state
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchResults, setBatchResults] = useState<Array<{
    file: File;
    preview: string;
    label: string;
    confidence: number;
    inferenceTime: number;
    status: "pending" | "processing" | "done" | "error";
  }>>([]);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [currentProcessingIndex, setCurrentProcessingIndex] = useState(-1);
  
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
        const sess = await ort.InferenceSession.create("/model/mobilenetv4.onnx", {
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

  // 수치적 안정성 고려한 Softmax
  const _softmax = (logits: Float32Array): Float32Array => {
    // 오버플로우 방지: 최대값을 빼줌
    const max = Math.max(...logits);
    // 각 값에 e^(x - max) 계산
    const expValues = Array.from(logits).map(x => Math.exp(x - max));
    // 정규화: 합으로 나눔
    const sum = expValues.reduce((a, b) => a + b, 0);
    
    return new Float32Array(expValues.map(x => x / sum));
  }

  // 핵심 추론 함수 - 단일/배치 모드에서 재사용
  const processImage = async (imageSrc: string): Promise<{
    label: string;
    confidence: number;
    classIndex: number;
  }> => {
    if (!session || labels.length === 0) {
      throw new Error("Model not ready");
    }

    // 1. 이미지 객체 생성 및 로드 대기
    const img = new window.Image();
    img.src = imageSrc;
    await new Promise((resolve) => (img.onload = resolve));

    // 2. 캔버스를 이용한 리사이징
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
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

    // // 5. 최댓값 찾기
    // let maxIdx = 0,
    //   maxVal = -Infinity;
    // for (let i = 0; i < output.length; i++) {
    //   if (output[i] > maxVal) {
    //     maxVal = output[i];
    //     maxIdx = i;
    //   }
    // }

    const probs = _softmax(output);
    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label: labels[maxIdx] || `Unknown (${maxIdx})`,
      confidence: probs[maxIdx], // 0~1 사이 확률
      classIndex: maxIdx,
    };
  };

  // 단일 이미지 추론 (기존 UI용)
  const runInference = async () => {
    if (!session || !image || labels.length === 0) return;
    setIsProcessing(true);
    setResult("분석 중...");

    try {
      const result = await processImage(image);
      setResult(`결과: ${result.label} (신뢰도: ${result.confidence.toFixed(2)})`);
    } catch (error) {
      console.error(error);
      setResult("에러 발생");
    } finally {
      setIsProcessing(false);
    }
  };

  // 배치처리용 코드
  const handleBatchUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
  const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setBatchFiles(files);
    setBatchResults(
      files.map((file) => ({
        file,
        preview: URL.createObjectURL(file),
        label: "",
        confidence: 0,
        inferenceTime: 0,
        status: "pending" as const,
      }))
    );
  };
  const runBatchInference = async () => {
    if (!session || batchFiles.length === 0 || labels.length === 0) return;
    
    setIsBatchProcessing(true);
    
    for (let i = 0; i < batchFiles.length; i++) {
      setCurrentProcessingIndex(i);
      
      // 상태를 "processing"으로 업데이트
      setBatchResults((prev) => {
        const updated = [...prev];
        updated[i] = { ...updated[i], status: "processing" };
        return updated;
      });
      try {
        const startTime = performance.now();
        
        // 공통 추론 함수 사용
        const result = await processImage(batchResults[i].preview);
        
        const endTime = performance.now();
        const inferenceTime = endTime - startTime;
        
        // 결과 업데이트
        setBatchResults((prev) => {
          const updated = [...prev];
          updated[i] = {
            ...updated[i],
            label: result.label,
            confidence: result.confidence,
            inferenceTime,
            status: "done",
          };
          return updated;
        });
      } catch (error) {
        console.error(`Error processing image ${i}:`, error);
        setBatchResults((prev) => {
          const updated = [...prev];
          updated[i] = { ...updated[i], status: "error" };
          return updated;
        });
      }
    }
    
    setIsBatchProcessing(false);
    setCurrentProcessingIndex(-1);
  };

  if (!mounted) return null;

  return (
    <main style={{ width: "100%", maxWidth: "500px", padding: "20px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "20px" }}>
      <header>
        <h1 style={{ fontSize: "24px", fontWeight: "800", margin: "0 0 8px 0" }}>OCR-TEST</h1>
        <p style={{ fontSize: "14px", color: "#666", margin: 0 }}>{loadingStatus}</p>
      </header>

      {isModelLoaded && session && (
        <div style={{ padding: "12px", backgroundColor: "#f0f7ff", color: "#0070f3", borderRadius: "10px", fontSize: "12px", fontWeight: "600" }}>
          ✅ 모델 준비 완료
        </div>
      )}
      {/* 탭 버튼 */}
      <div style={{ display: "flex", gap: "10px", borderBottom: "2px solid #e5e7eb" }}>
        <button
          onClick={() => setActiveTab("single")}
          style={{
            padding: "12px 24px",
            border: "none",
            backgroundColor: "transparent",
            borderBottom: activeTab === "single" ? "3px solid #0070f3" : "none",
            color: activeTab === "single" ? "#0070f3" : "#666",
            fontWeight: "600",
            cursor: "pointer",
          }}
        >
          단일 이미지
        </button>
        <button
          onClick={() => setActiveTab("batch")}
          style={{
            padding: "12px 24px",
            border: "none",
            backgroundColor: "transparent",
            borderBottom: activeTab === "batch" ? "3px solid #0070f3" : "none",
            color: activeTab === "batch" ? "#0070f3" : "#666",
            fontWeight: "600",
            cursor: "pointer",
          }}
        >
          배치 처리
        </button>
      </div>
      {/* 단일 이미지 모드 (기존 코드 그대로) */}
      {activeTab === "single" && (
        <>
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
        </>
      )}
      {/* 배치 처리 모드 */}
      {activeTab === "batch" && (
        <>
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleBatchUpload}
              style={{ padding: "12px", border: "2px solid #e5e7eb", borderRadius: "10px" }}
            />
            
            {batchFiles.length > 0 && (
              <div style={{ padding: "12px", backgroundColor: "#f0f7ff", borderRadius: "10px", color: "black" }}>
                <p style={{ margin: 0, fontSize: "14px", fontWeight: "600" }}>
                  {batchFiles.length}개 파일 선택됨
                </p>
              </div>
            )}
            {batchFiles.length > 0 && isModelLoaded && !isBatchProcessing && (
              <button
                onClick={runBatchInference}
                style={{ width: "100%", padding: "16px", borderRadius: "14px", backgroundColor: "#0070f3", color: "white", border: "none", fontWeight: "600", cursor: "pointer" }}
              >
                모두 분석하기
              </button>
            )}
            {isBatchProcessing && (
              <div style={{ padding: "12px", backgroundColor: "#fef3c7", borderRadius: "10px" }}>
                <p style={{ margin: 0, fontSize: "14px", fontWeight: "600" }}>
                  처리 중: {currentProcessingIndex + 1} / {batchFiles.length}
                </p>
              </div>
            )}
            {/* 결과 리스트 */}
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {batchResults.map((result, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "flex",
                    gap: "16px",
                    padding: "16px",
                    border: "1px solid #e5e7eb",
                    borderRadius: "12px",
                    backgroundColor: result.status === "done" ? "#f0fdf4" : result.status === "error" ? "#fef2f2" : "#f9fafb",
                  }}
                >
                  {/* 썸네일 */}
                  <div style={{ position: "relative", width: "80px", height: "80px", flexShrink: 0, borderRadius: "8px", overflow: "hidden" }}>
                    <Image src={result.preview} alt={result.file.name} fill style={{ objectFit: "cover" }} unoptimized />
                  </div>
                  {/* 정보 */}
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
                    <p style={{ margin: 0, fontSize: "14px", fontWeight: "600", color: "#111" }}>
                      {result.file.name}
                    </p>
                    
                    {result.status === "done" && (
                      <>
                        <p style={{ margin: 0, fontSize: "13px", color: "#0070f3" }}>
                          🏷️ {result.label}
                        </p>
                        <p style={{ margin: 0, fontSize: "12px", color: "#666" }}>
                          신뢰도: {(result.confidence * 100).toFixed(2)}%
                        </p>
                        <p style={{ margin: 0, fontSize: "12px", color: "#666" }}>
                          추론 시간: {result.inferenceTime.toFixed(0)}ms
                        </p>
                      </>
                    )}
                    
                    {result.status === "processing" && (
                      <p style={{ margin: 0, fontSize: "13px", color: "#f59e0b" }}>⏳ 처리 중...</p>
                    )}
                    
                    {result.status === "pending" && (
                      <p style={{ margin: 0, fontSize: "13px", color: "#9ca3af" }}>⏸️ 대기 중</p>
                    )}
                    
                    {result.status === "error" && (
                      <p style={{ margin: 0, fontSize: "13px", color: "#dc2626" }}>❌ 오류 발생</p>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* 통계 표 */}
            {batchResults.length > 0 && batchResults.some(r => r.status === "done") && (
              <div style={{ marginTop: "24px", padding: "20px", backgroundColor: "#f9fafb", borderRadius: "16px", border: "1px solid #e5e7eb" }}>
                <h3 style={{ margin: "0 0 16px 0", fontSize: "18px", fontWeight: "700", color: "#111" }}>
                  📊 배치 처리 통계
                </h3>
                
                {(() => {
                  const doneResults = batchResults.filter(r => r.status === "done");
                  if (doneResults.length === 0) return null;

                  // 라벨별 카운트
                  const labelCounts: Record<string, number> = {};
                  doneResults.forEach(r => {
                    labelCounts[r.label] = (labelCounts[r.label] || 0) + 1;
                  });

                  // 평균 계산
                  const avgConfidence = doneResults.reduce((sum, r) => sum + r.confidence, 0) / doneResults.length;
                  const avgInferenceTime = doneResults.reduce((sum, r) => sum + r.inferenceTime, 0) / doneResults.length;

                  // 최고/최저 추론 시간
                  const fastest = doneResults.reduce((min, r) => r.inferenceTime < min.inferenceTime ? r : min);
                  const slowest = doneResults.reduce((max, r) => r.inferenceTime > max.inferenceTime ? r : max);

                  return (
                    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                      {/* 라벨 분포 */}
                      <div style={{ padding: "16px", backgroundColor: "white", borderRadius: "12px" }}>
                        <h4 style={{ margin: "0 0 12px 0", fontSize: "14px", fontWeight: "600", color: "#666" }}>
                          🏷️ 분류 결과 분포
                        </h4>
                        <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                          {Object.entries(labelCounts).map(([label, count]) => (
                            <div key={label} style={{ padding: "8px 16px", backgroundColor: "#f0f7ff", borderRadius: "8px" }}>
                              <span style={{ fontSize: "13px", fontWeight: "600", color: "#0070f3" }}>
                                {label}: {count}개 ({((count / doneResults.length) * 100).toFixed(1)}%)
                              </span>
                            </div>
                          ))}
                        </div>
                        <p style={{ margin: "8px 0 0 0", fontSize: "12px", color: "#666" }}>
                          총 {doneResults.length}개 처리 완료
                        </p>
                      </div>

                      {/* 통계 테이블 */}
                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", backgroundColor: "white", borderRadius: "12px", overflow: "hidden" }}>
                          <thead>
                            <tr style={{ backgroundColor: "#f3f4f6" }}>
                              <th style={{ padding: "12px", textAlign: "left", fontSize: "13px", fontWeight: "600", color: "#374151", borderBottom: "2px solid #e5e7eb" }}>
                                항목
                              </th>
                              <th style={{ padding: "12px", textAlign: "right", fontSize: "13px", fontWeight: "600", color: "#374151", borderBottom: "2px solid #e5e7eb" }}>
                                값
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#111", borderBottom: "1px solid #f3f4f6" }}>
                                평균 신뢰도
                              </td>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#0070f3", fontWeight: "600", textAlign: "right", borderBottom: "1px solid #f3f4f6" }}>
                                {(avgConfidence * 100).toFixed(2)}%
                              </td>
                            </tr>
                            <tr>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#111", borderBottom: "1px solid #f3f4f6" }}>
                                평균 추론 시간
                              </td>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#0070f3", fontWeight: "600", textAlign: "right", borderBottom: "1px solid #f3f4f6" }}>
                                {avgInferenceTime.toFixed(0)}ms
                              </td>
                            </tr>
                            <tr>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#111", borderBottom: "1px solid #f3f4f6" }}>
                                가장 빠른 추론
                              </td>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#10b981", fontWeight: "600", textAlign: "right", borderBottom: "1px solid #f3f4f6" }}>
                                {fastest.inferenceTime.toFixed(0)}ms
                                <div style={{ fontSize: "11px", color: "#666", fontWeight: "400", marginTop: "2px" }}>
                                  {fastest.file.name}
                                </div>
                              </td>
                            </tr>
                            <tr>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#111" }}>
                                가장 느린 추론
                              </td>
                              <td style={{ padding: "12px", fontSize: "13px", color: "#ef4444", fontWeight: "600", textAlign: "right" }}>
                                {slowest.inferenceTime.toFixed(0)}ms
                                <div style={{ fontSize: "11px", color: "#666", fontWeight: "400", marginTop: "2px" }}>
                                  {slowest.file.name}
                                </div>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </>
      )}
    </main>
  );
}
