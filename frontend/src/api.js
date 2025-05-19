import * as FileSystem from "expo-file-system";

const BASE_URL = "http://192.168.31.144:8000";

export async function predictSymbol(base64) {
  try {
    // Geçici dosya yolu oluştur
    const tmpPath = FileSystem.cacheDirectory + "symbol.png";

    // Remove data URI prefix if it exists
    let imageData = base64;
    if (base64.startsWith("data:image/png;base64,")) {
      imageData = base64.split(",")[1];
    }

    // Base64 verisini doğrudan decode ederek dosyaya yaz
    await FileSystem.writeAsStringAsync(tmpPath, imageData, {
      encoding: FileSystem.EncodingType.Base64,
    });

    // FormData ile dosyayı gönder
    const formData = new FormData();

    // Read file info
    const fileInfo = await FileSystem.getInfoAsync(tmpPath);

    formData.append("file", {
      uri: fileInfo.uri,
      name: "symbol.png",
      type: "image/png",
    });

    // Gönderim (başlık ekleme!)
    const res = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "multipart/form-data",
      },
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}, ${await res.text()}`);
    }
    return await res.json();
  } catch (err) {
    console.error("predictSymbol error:", err);
    throw err;
  }
}
