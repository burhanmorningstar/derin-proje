import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Dimensions,
} from 'react-native';
import { Video, ResizeMode } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as ImageManipulator from 'expo-image-manipulator';

const { width, height } = Dimensions.get('window');

// API endpoint URL'inizi buraya yazın
const API_ENDPOINT = 'http://192.168.1.110:5000/predict';

export default function App() {
  const [videoUri, setVideoUri] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [intervalId, setIntervalId] = useState(null);
  
  const videoRef = useRef(null);

  // Video seçme fonksiyonu
  const selectVideo = async () => {
    try {
      // Medya kütüphanesi izinlerini kontrol et
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Hata', 'Galeri erişim izni gerekli!');
        return;
      }

      // Video seçme
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });
      

      if (!result.canceled && result.assets[0]) {
        setVideoUri(result.assets[0].uri);
        setPredictions([]);
        
        // Eğer önceki interval varsa temizle
        if (intervalId) {
          clearInterval(intervalId);
          setIntervalId(null);
        }
      }
    } catch (error) {
      console.error('Video seçme hatası:', error);
      Alert.alert('Hata', 'Video seçilirken bir hata oluştu');
    }
  };

  // Video frame'ini yakala ve API'ye gönder
  const captureAndSendFrame = async () => {
    if (!videoUri || !videoRef.current) return;

    try {
      setIsLoading(true);

      // Video'dan mevcut pozisyonda thumbnail oluştur
      const status = await videoRef.current.getStatusAsync();
      const currentTime = status.positionMillis || 0;
      
      const { uri: thumbnailUri } = await VideoThumbnails.getThumbnailAsync(
        videoUri,
        {
          time: currentTime,
          quality: 0.7,
        }
      );

      // Thumbnail'i optimize et
      const manipResult = await ImageManipulator.manipulateAsync(
        thumbnailUri,
        [{ resize: { width: 640 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );

      // Optimize edilmiş görüntüyü API'ye gönder
      const formData = new FormData();
      formData.append('frames', {
        uri: manipResult.uri,
        type: 'image/jpeg',
        name: 'frame.jpg',
      });

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // Başarılı prediction'ı listeye ekle
      const newPrediction = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString('tr-TR'),
        result: result,
      };
      
      setPredictions(prev => [newPrediction, ...prev.slice(0, 19)]); // Son 20 sonucu tut
    } catch (error) {
      console.error('Frame gönderme hatası:', error);
      Alert.alert('Hata', 'Frame gönderilirken hata oluştu: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Video oynatmayı başlat ve periyodik API çağrılarını başlat
  const startPlayback = async () => {
    if (!videoUri) {
      Alert.alert('Hata', 'Önce bir video seçin!');
      return;
    }

    try {
      // Video oynatmayı başlat
      await videoRef.current?.playAsync();
      setIsPlaying(true);

      // Her 2 saniyede bir API çağrısı yap
      const id = setInterval(captureAndSendFrame, 2000);
      setIntervalId(id);

      // İlk frame'i hemen gönder
      captureAndSendFrame();
    } catch (error) {
      console.error('Oynatma hatası:', error);
      Alert.alert('Hata', 'Video oynatılırken hata oluştu');
    }
  };

  // Video oynatmayı durdur
  const stopPlayback = async () => {
    try {
      await videoRef.current?.pauseAsync();
      setIsPlaying(false);

      // Interval'i temizle
      if (intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }
    } catch (error) {
      console.error('Durdurma hatası:', error);
    }
  };

  // Component unmount olduğunda interval'i temizle
  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Video Prediction App</Text>
      </View>

      {/* Video Player */}
      {videoUri ? (
        <View style={styles.videoContainer}>
          <Video
            ref={videoRef}
            source={{ uri: videoUri }}
            style={styles.video}
            useNativeControls
            resizeMode={ResizeMode.CONTAIN}
            shouldPlay={false}
            isLooping
          />
        </View>
      ) : (
        <View style={styles.placeholderContainer}>
          <Text style={styles.placeholderText}>Video seçin</Text>
        </View>
      )}

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity style={styles.button} onPress={selectVideo}>
          <Text style={styles.buttonText}>Video Seç</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, !videoUri && styles.buttonDisabled]}
          onPress={isPlaying ? stopPlayback : startPlayback}
          disabled={!videoUri}
        >
          <Text style={styles.buttonText}>
            {isPlaying ? 'Durdur' : 'Başlat'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Loading Indicator */}
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.loadingText}>API'ye gönderiliyor...</Text>
        </View>
      )}

      {/* Predictions List */}
      <View style={styles.predictionsContainer}>
        <Text style={styles.predictionsTitle}>
          Prediction Sonuçları ({predictions.length})
        </Text>
        
        <ScrollView style={styles.predictionsList}>
          {predictions.map((prediction) => (
            <View key={prediction.id} style={styles.predictionItem}>
              <Text style={styles.predictionTime}>{prediction.timestamp}</Text>
              <Text style={styles.predictionResult}>
                {JSON.stringify(prediction.result, null, 2)}
              </Text>
            </View>
          ))}
          
          {predictions.length === 0 && (
            <Text style={styles.noPredictions}>
              Henüz prediction yok. Video başlatın!
            </Text>
          )}
        </ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    paddingTop: 50,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#007AFF',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
  videoContainer: {
    height: 200,
    margin: 20,
    backgroundColor: 'black',
    borderRadius: 10,
    overflow: 'hidden',
  },
  video: {
    flex: 1,
  },
  placeholderContainer: {
    height: 200,
    margin: 20,
    backgroundColor: '#e0e0e0',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    minWidth: 100,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 10,
  },
  loadingText: {
    marginLeft: 10,
    fontSize: 14,
    color: '#666',
  },
  predictionsContainer: {
    flex: 1,
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
  },
  predictionsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  predictionsList: {
    flex: 1,
  },
  predictionItem: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    marginBottom: 8,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#007AFF',
  },
  predictionTime: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  predictionResult: {
    fontSize: 14,
    color: '#333',
    fontFamily: 'monospace',
  },
  noPredictions: {
    textAlign: 'center',
    color: '#666',
    fontStyle: 'italic',
    marginTop: 20,
  },
});