// Performance and memory management utilities

// Image memory cleanup
export const cleanupImageUri = (uri) => {
  if (uri && uri.startsWith("file://")) {
    // For local files, we can't manually clean them up
    // but we can mark them for garbage collection
    return null;
  }
  return uri;
};

// Memory usage tracking
export const getMemoryInfo = () => {
  if (__DEV__ && global.performance && global.performance.memory) {
    return {
      usedJSHeapSize: global.performance.memory.usedJSHeapSize,
      totalJSHeapSize: global.performance.memory.totalJSHeapSize,
      jsHeapSizeLimit: global.performance.memory.jsHeapSizeLimit,
    };
  }
  return null;
};

// Frame rate monitoring
export const createFPSMonitor = () => {
  let frames = 0;
  let lastTime = performance.now();
  let fps = 0;

  const update = () => {
    frames++;
    const currentTime = performance.now();

    if (currentTime >= lastTime + 1000) {
      fps = Math.round((frames * 1000) / (currentTime - lastTime));
      frames = 0;
      lastTime = currentTime;
    }

    requestAnimationFrame(update);
  };

  if (__DEV__) {
    requestAnimationFrame(update);
  }

  return () => fps;
};

// Batch updates for better performance
export const batchUpdates = (callback) => {
  if (global.requestIdleCallback) {
    global.requestIdleCallback(callback);
  } else {
    setTimeout(callback, 0);
  }
};

// Image size optimization
export const calculateOptimalImageSize = (
  originalWidth,
  originalHeight,
  maxSize = 1024
) => {
  const aspectRatio = originalWidth / originalHeight;

  if (originalWidth <= maxSize && originalHeight <= maxSize) {
    return { width: originalWidth, height: originalHeight };
  }

  if (aspectRatio > 1) {
    // Landscape
    return {
      width: maxSize,
      height: Math.round(maxSize / aspectRatio),
    };
  } else {
    // Portrait
    return {
      width: Math.round(maxSize * aspectRatio),
      height: maxSize,
    };
  }
};

// Network status utilities
export const getNetworkType = async () => {
  try {
    if (global.navigator && global.navigator.connection) {
      const connection = global.navigator.connection;
      return {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
        saveData: connection.saveData,
      };
    }
  } catch (error) {
    console.warn("Network API not supported:", error);
  }
  return null;
};
