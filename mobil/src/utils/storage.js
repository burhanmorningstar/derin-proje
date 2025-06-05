import AsyncStorage from "@react-native-async-storage/async-storage";

// Storage keys
const STORAGE_KEYS = {
  PREDICTIONS_HISTORY: "@predictions_history",
  APP_SETTINGS: "@app_settings",
  USER_PREFERENCES: "@user_preferences",
};

// Storage utility class
class StorageManager {
  // Save predictions history
  async savePredictionsHistory(predictions) {
    try {
      const jsonValue = JSON.stringify(predictions);
      await AsyncStorage.setItem(STORAGE_KEYS.PREDICTIONS_HISTORY, jsonValue);
    } catch (error) {
      console.error("Error saving predictions history:", error);
    }
  }

  // Load predictions history
  async loadPredictionsHistory() {
    try {
      const jsonValue = await AsyncStorage.getItem(
        STORAGE_KEYS.PREDICTIONS_HISTORY
      );
      return jsonValue != null ? JSON.parse(jsonValue) : [];
    } catch (error) {
      console.error("Error loading predictions history:", error);
      return [];
    }
  }

  // Save app settings
  async saveAppSettings(settings) {
    try {
      const jsonValue = JSON.stringify(settings);
      await AsyncStorage.setItem(STORAGE_KEYS.APP_SETTINGS, jsonValue);
    } catch (error) {
      console.error("Error saving app settings:", error);
    }
  }

  // Load app settings
  async loadAppSettings() {
    try {
      const jsonValue = await AsyncStorage.getItem(STORAGE_KEYS.APP_SETTINGS);
      return jsonValue != null ? JSON.parse(jsonValue) : {};
    } catch (error) {
      console.error("Error loading app settings:", error);
      return {};
    }
  }

  // Clear all data
  async clearAllData() {
    try {
      await AsyncStorage.multiRemove([
        STORAGE_KEYS.PREDICTIONS_HISTORY,
        STORAGE_KEYS.APP_SETTINGS,
        STORAGE_KEYS.USER_PREFERENCES,
      ]);
    } catch (error) {
      console.error("Error clearing data:", error);
    }
  }

  // Get storage info
  async getStorageInfo() {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const stores = await AsyncStorage.multiGet(keys);
      return stores.map(([key, value]) => ({
        key,
        size: value ? value.length : 0,
      }));
    } catch (error) {
      console.error("Error getting storage info:", error);
      return [];
    }
  }
}

export default new StorageManager();
