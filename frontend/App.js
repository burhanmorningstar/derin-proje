import React, { useState } from "react";
import { SafeAreaView, Text, ActivityIndicator, Alert } from "react-native";
import DrawingCanvas from "./src/DrawingCanvas";
import { predictSymbol } from "./src/api";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handlePredict(base64) {
    setLoading(true);
    setResult(null);
    try {
      const res = await predictSymbol(base64);
      setResult(res); // {symbol, confidence}
    } catch (err) {
      Alert.alert("Error", err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <DrawingCanvas onPredict={handlePredict} />
      {loading && (
        <ActivityIndicator
          style={{ position: "absolute", top: "50%", left: "45%" }}
        />
      )}
      {result && (
        <Text
          style={{
            position: "absolute",
            top: 40,
            alignSelf: "center",
            fontSize: 22,
          }}
        >
          {result.symbol} ({(result.confidence * 100).toFixed(1)}%)
        </Text>
      )}
    </SafeAreaView>
  );
}
