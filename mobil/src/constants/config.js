// Application configuration constants
export const CONFIG = {
  API: {
    BASE_URL: "http://192.168.68.114:5000",
    ENDPOINTS: {
      PREDICT: "/predict",
    },
    TIMEOUT: 30000, // 30 seconds
  },
  VIDEO: {
    FRAME_CAPTURE_INTERVAL: 2000, // 2 seconds
    THUMBNAIL_QUALITY: 0.7,
    THUMBNAIL_WIDTH: 640,
    COMPRESSION_QUALITY: 0.8,
  },

  UI: {
    COLORS: {
      PRIMARY: "#007AFF",
      SECONDARY: "#5856D6",
      SUCCESS: "#28A745",
      WARNING: "#FFC107",
      DANGER: "#DC3545",
      VIOLENCE: "#D90429",
      NON_VIOLENCE: "#23C552",
      BACKGROUND: "#000000",
      CARD_BACKGROUND: "#FFFFFF",
      TEXT_PRIMARY: "#212529",
      TEXT_SECONDARY: "#6C757D",
      BORDER: "#DEE2E6",
      SHADOW: "rgba(0, 0, 0, 0.1)",
      HEADER: "#FFA500",
    },
    SPACING: {
      XS: 2,
      SM: 4,
      MD: 8,
      LG: 12,
      XL: 16,
    },
    BORDER_RADIUS: {
      SM: 4,
      MD: 8,
      LG: 12,
      XL: 16,
    },
    FONT_SIZES: {
      XS: 11,
      SM: 12,
      MD: 14,
      LG: 16,
      XL: 18,
      XXL: 20,
      XXXL: 24,
    },
  },
};

export const MESSAGES = {
  ERRORS: {
    PERMISSION_DENIED: "Galeri erişim izni gerekli!",
    VIDEO_SELECTION: "Video seçilirken bir hata oluştu",
    VIDEO_PLAYBACK: "Video oynatılırken hata oluştu",
    FRAME_CAPTURE: "Frame gönderilirken hata oluştu",
    NETWORK_ERROR: "Ağ bağlantısında sorun var",
    NO_VIDEO_SELECTED: "Önce bir video seçin!",
  },
  SUCCESS: {
    VIDEO_SELECTED: "Video başarıyla seçildi",
    PREDICTION_SUCCESS: "Tahmin başarıyla alındı",
  },
  INFO: {
    SELECT_VIDEO: "Video seçin",
    NO_PREDICTIONS: "Henüz prediction yok. Video başlatın!",
    SENDING_TO_API: "Görüntü analiz ediliyor...",
    PREDICTIONS_LIMIT_WARNING: "Tahmin sınırına yaklaşıyorsunuz",
  },
};
