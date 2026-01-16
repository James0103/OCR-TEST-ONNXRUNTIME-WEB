"use client";

import React, { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "/wasm/";

export default function Home() {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("준비 중...");
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  // 실행 코드
  useEffect(() => {
    async function loadModel() {
      try {
        setLoadingStatus("AI 모델을 불러오는 중...");
        // 1. 모델 세션 생성
        // executionProviders를 'wasm'으로 지정하여 브라우저 로컬 엔진을 사용합니다.
        const sess = await ort.InferenceSession.create("/model/mobilenetv2.onnx", {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        setSession(sess);
        setIsModelLoaded(true);
        setLoadingStatus("모델 준비 완료!");
        console.log("ONNX Model Loaded successfully:", sess);
      } catch (error) {
        console.error("모델 로딩 실패:", error);
        setLoadingStatus("모델 로딩에 실패했습니다. (콘솔 확인)");
      }
    }

    // 코드 실행(useEffect)
    // page onMounted 시
    loadModel();
  }, []);

  // UI 설계
  return (
    <main style={{
      width: "100%",
      maxWidth: "500px",
      padding: "20px",
      display: "flex",
      flexDirection: "column",
      gap: "20px"
    }}>
      {/* 헤더 */}
      <header style={{ textAlign: "left", marginBottom: "10px" }}>
        <h1 style={{ fontSize: "24px", fontWeight: "800", margin: "0 0 8px 0" }}>Vision AI</h1>
        <p style={{ fontSize: "14px", color: "#666", margin: 0 }}>{loadingStatus}</p>
      </header>
      {/* 헤더 */}

      {/* 로딩 표시기 (간단한 디자인) */}
      {!isModelLoaded && (
        <div style={{
          width: "100%",
          height: "4px",
          backgroundColor: "#eee",
          borderRadius: "2px",
          overflow: "hidden"
        }}>
          <div style={{
            width: "50%",
            height: "100%",
            backgroundColor: "#0070f3",
            animation: "loading-bar 1.5s infinite ease-in-out"
          }} />
        </div>
      )}

    </main>
  );
}
