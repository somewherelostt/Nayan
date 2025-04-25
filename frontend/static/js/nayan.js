/**
 * Nayan - AI Vision Assistant
 * Frontend JavaScript
 */

document.addEventListener("DOMContentLoaded", function () {
  // UI Elements
  const announcements = document.getElementById("announcements");
  const alerts = document.getElementById("alerts");
  const sceneTypeEl = document.getElementById("scene-type");
  const objectCountEl = document.getElementById("object-count");
  const peopleCountEl = document.getElementById("people-count");
  const helpButton = document.getElementById("help-button");
  const commandPanel = document.getElementById("command-panel");
  const closeCommands = document.getElementById("close-commands");

  // Control buttons
  const btnScene = document.getElementById("btn-scene");
  const btnObjects = document.getElementById("btn-objects");
  const btnText = document.getElementById("btn-text");
  const btnNavigate = document.getElementById("btn-navigate");
  const btnToggleMode = document.getElementById("btn-toggle-mode");

  // Audio elements
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const soundCache = {};

  // Speech synthesis
  const speechSynthesis = window.speechSynthesis;

  // WebSocket connection
  let socket;

  // Check if device is mobile
  const isMobile =
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent
    );

  // Set lightweight mode by default on mobile devices
  if (isMobile) {
    console.log("Mobile device detected, optimizing performance");
    // Will toggle to lightweight mode after connection
    setTimeout(() => {
      togglePerformanceMode();
    }, 2000);
  }

  // Initialize
  init();

  // Initialize the application
  function init() {
    // Add event listeners to buttons
    btnScene.addEventListener("click", describeScene);
    btnObjects.addEventListener("click", identifyObjects);
    btnText.addEventListener("click", readText);
    btnNavigate.addEventListener("click", navigate);
    btnToggleMode.addEventListener("click", togglePerformanceMode);

    // Help button and command panel
    helpButton.addEventListener("click", toggleCommandPanel);
    closeCommands.addEventListener("click", toggleCommandPanel);

    // Initialize WebSocket
    initWebSocket();

    // Preload sound effects
    preloadSounds();

    // Simulate some initial data
    setTimeout(() => {
      updateEnvironmentInfo("indoors", 5, 1);
      addAlert("System initialized and ready for use.", "info");
    }, 1000);
  }

  // Initialize WebSocket connection
  function initWebSocket() {
    // Check if SocketIO is available
    if (typeof io === "undefined") {
      console.error("Socket.IO not loaded");
      addAlert(
        "Real-time updates unavailable - reload the page to try again",
        "danger"
      );
      loadSocketIOScript();
      return;
    }

    // Connect to the server
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}`;

    console.log(`Connecting to WebSocket at ${wsUrl}`);
    socket = io(wsUrl);

    // Connection events
    socket.on("connect", () => {
      console.log("Connected to server");
      addAlert("Connected to Nayan server", "success");
      speakText("Nayan Vision Assistant ready");

      // Update connection status
      updateConnectionStatus(true);

      // Auto-switch to lightweight mode on mobile
      if (isMobile && !window.lightweightModeEnabled) {
        console.log("Auto-switching to lightweight mode for mobile");
        setTimeout(() => {
          togglePerformanceMode();
        }, 2000);
      }
    });

    socket.on("disconnect", () => {
      console.log("Disconnected from server");
      addAlert("Disconnected from server", "warning");
      updateConnectionStatus(false);
    });

    // Custom events
    socket.on("announcement", (data) => {
      console.log("Received announcement:", data);
      addAnnouncement(data.text);
      speakText(data.text, data.priority);
    });

    socket.on("sound", (data) => {
      console.log("Received sound trigger:", data);
      playSound(data.name);
    });
  }

  // Dynamically load Socket.IO script if needed
  function loadSocketIOScript() {
    const script = document.createElement("script");
    script.src = "https://cdn.socket.io/4.6.0/socket.io.min.js";
    script.onload = function () {
      console.log("Socket.IO script loaded");
      initWebSocket();
    };
    script.onerror = function () {
      console.error("Failed to load Socket.IO script");
    };
    document.head.appendChild(script);
  }

  // Preload common sounds
  function preloadSounds() {
    // Define sounds to preload
    const soundsToLoad = [
      { name: "proximity_alert", url: "/sounds/proximity_alert.mp3" },
      { name: "object_detected", url: "/sounds/object_detected.mp3" },
    ];

    // Create fallback sounds in case the files aren't available
    createFallbackSounds();

    // Load each sound
    soundsToLoad.forEach((sound) => {
      fetch(sound.url)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.arrayBuffer();
        })
        .then((arrayBuffer) => {
          // Decode the audio data
          return audioContext.decodeAudioData(arrayBuffer);
        })
        .then((audioBuffer) => {
          // Store in the cache
          soundCache[sound.name] = audioBuffer;
          console.log(`Sound loaded: ${sound.name}`);
        })
        .catch((error) => {
          console.error(`Could not load sound ${sound.name}:`, error);
          // Use fallback beep sounds
          useFallbackSound(sound.name);
        });
    });
  }

  // Create fallback sounds using oscillator
  function createFallbackSounds() {
    // Create beep sounds of different frequencies
    const createBeep = (frequency, duration) => {
      const sampleRate = audioContext.sampleRate;
      const numFrames = duration * sampleRate;
      const buffer = audioContext.createBuffer(1, numFrames, sampleRate);
      const data = buffer.getChannelData(0);

      for (let i = 0; i < numFrames; i++) {
        const t = i / sampleRate;
        // Simple sine wave
        data[i] = Math.sin(2 * Math.PI * frequency * t) * (1 - t / duration); // Add fade out
      }

      return buffer;
    };

    // Create fallback sounds
    soundCache["fallback_alert"] = createBeep(880, 0.3); // A5 note
    soundCache["fallback_notification"] = createBeep(440, 0.2); // A4 note
  }

  // Use fallback sound when original can't be loaded
  function useFallbackSound(soundName) {
    if (soundName.includes("alert")) {
      soundCache[soundName] = soundCache["fallback_alert"];
    } else {
      soundCache[soundName] = soundCache["fallback_notification"];
    }
  }

  // Play a sound
  function playSound(soundName) {
    try {
      // Check if sound exists in cache
      if (!soundCache[soundName]) {
        console.warn(`Sound ${soundName} not loaded, using fallback`);
        // Use fallback sound
        soundName = soundName.includes("alert")
          ? "fallback_alert"
          : "fallback_notification";
      }

      // Get buffer from cache
      const buffer = soundCache[soundName];

      // Create source node
      const source = audioContext.createBufferSource();
      source.buffer = buffer;

      // Connect to output
      source.connect(audioContext.destination);

      // Play sound
      source.start(0);

      console.log(`Playing sound: ${soundName}`);
    } catch (error) {
      console.error("Error playing sound:", error);
    }
  }

  // Speak text using Web Speech API
  function speakText(text, priority = false) {
    if (!text || !speechSynthesis) return;

    // Cancel current speech if this is a priority message
    if (priority && speechSynthesis.speaking) {
      speechSynthesis.cancel();
    }

    // Create utterance
    const utterance = new SpeechSynthesisUtterance(text);

    // Set properties
    utterance.rate = 1.0; // Speed
    utterance.pitch = 1.0; // Pitch
    utterance.volume = 1.0; // Volume

    // Try to use a female voice if available
    const voices = speechSynthesis.getVoices();
    if (voices.length > 0) {
      // Prefer English female voice
      const femaleVoice = voices.find(
        (voice) => voice.name.includes("Female") && voice.lang.includes("en")
      );
      if (femaleVoice) {
        utterance.voice = femaleVoice;
      }
    }

    // Speak the text
    speechSynthesis.speak(utterance);
  }

  // Toggle command panel visibility
  function toggleCommandPanel() {
    commandPanel.classList.toggle("active");
  }

  // Add an announcement to the UI
  function addAnnouncement(message) {
    const announcement = document.createElement("div");
    announcement.className = "announcement";
    announcement.innerHTML = `<p>${message}</p>`;

    // Add to the beginning of the list
    if (announcements.firstChild) {
      announcements.insertBefore(announcement, announcements.firstChild);
    } else {
      announcements.appendChild(announcement);
    }

    // Remove old announcements if there are too many
    const maxAnnouncements = 5;
    while (announcements.childElementCount > maxAnnouncements) {
      announcements.removeChild(announcements.lastChild);
    }

    // Scroll to the top of the announcements container
    announcements.scrollTop = 0;
  }

  // Add an alert to the UI
  function addAlert(message, type = "info") {
    const alert = document.createElement("div");
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `<p>${message}</p>`;

    // Add to the beginning of the list
    if (alerts.firstChild) {
      alerts.insertBefore(alert, alerts.firstChild);
    } else {
      alerts.appendChild(alert);
    }

    // Remove old alerts if there are too many
    const maxAlerts = 3;
    while (alerts.childElementCount > maxAlerts) {
      alerts.removeChild(alerts.lastChild);
    }

    // Scroll to the top of the alerts container
    alerts.scrollTop = 0;

    // Auto-remove after some time
    if (type !== "danger") {
      setTimeout(() => {
        if (alert.parentNode === alerts) {
          alert.style.opacity = "0";
          alert.style.transform = "translateY(-10px)";
          alert.style.transition = "opacity 0.5s, transform 0.5s";

          setTimeout(() => {
            if (alert.parentNode === alerts) {
              alerts.removeChild(alert);
            }
          }, 500);
        }
      }, 5000); // 5 seconds
    }
  }

  // Update environment information
  function updateEnvironmentInfo(sceneType, objectCount, peopleCount) {
    // Update with animation
    updateElementWithAnimation(sceneTypeEl, sceneType);
    updateElementWithAnimation(objectCountEl, objectCount);
    updateElementWithAnimation(peopleCountEl, peopleCount);
  }

  // Update element with animation
  function updateElementWithAnimation(element, value) {
    element.style.transform = "scale(1.1)";
    element.style.transition = "transform 0.3s";

    setTimeout(() => {
      element.textContent = value;
      element.style.transform = "scale(1)";
    }, 300);
  }

  // API Calls

  // Describe the current scene
  function describeScene() {
    addActiveState(btnScene);

    fetch("/api/scene_description")
      .then((response) => response.json())
      .then((data) => {
        addAnnouncement(data.description);

        // Extract and update environment info
        if (data.description.includes("indoors")) {
          updateEnvironmentInfo("indoors", "5+", "1+");
        } else if (data.description.includes("outdoors")) {
          updateEnvironmentInfo("outdoors", "3+", "2+");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        addAlert("Failed to get scene description");
      })
      .finally(() => {
        removeActiveState(btnScene);
      });
  }

  // Identify objects in the scene
  function identifyObjects() {
    addActiveState(btnObjects);

    fetch("/api/identify_objects")
      .then((response) => response.json())
      .then((data) => {
        addAnnouncement(data.objects);

        // Update object count (simple parse from the response)
        const objectText = data.objects;
        const objectMatches = objectText.match(/\d+/g);
        if (objectMatches && objectMatches.length > 0) {
          // Sum the numbers found
          const sum = objectMatches.reduce(
            (a, b) => parseInt(a) + parseInt(b),
            0
          );
          updateEnvironmentInfo("indoors", sum, "1+");
        }

        // Check for people
        if (objectText.includes("person") || objectText.includes("people")) {
          const peopleMatch = objectText.match(/(\d+)\s+(person|people)/);
          if (peopleMatch && peopleMatch.length > 1) {
            updateEnvironmentInfo("indoors", "5+", peopleMatch[1]);
          }
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        addAlert("Failed to identify objects");
      })
      .finally(() => {
        removeActiveState(btnObjects);
      });
  }

  // Read text in the scene
  function readText() {
    addActiveState(btnText);

    fetch("/api/read_text")
      .then((response) => response.json())
      .then((data) => {
        addAnnouncement("Looking for text to read...");
        addAlert("OCR activated. Results will be announced when found.");
      })
      .catch((error) => {
        console.error("Error:", error);
        addAlert("Failed to activate OCR");
      })
      .finally(() => {
        removeActiveState(btnText);
      });
  }

  // Get navigation guidance
  function navigate() {
    addActiveState(btnNavigate);

    fetch("/api/navigate")
      .then((response) => response.json())
      .then((data) => {
        addAnnouncement(data.guidance);

        // Play navigation sound
        playSound("proximity_alert");

        // Add alert based on guidance content
        if (data.guidance.includes("clear")) {
          addAlert("Path is clear ahead");
        } else if (data.guidance.includes("caution")) {
          addAlert("Proceed with caution");
        } else if (data.guidance.includes("Caution!")) {
          addAlert("Obstacles detected in all directions");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        addAlert("Failed to get navigation guidance");
      })
      .finally(() => {
        removeActiveState(btnNavigate);
      });
  }

  // Toggle between performance modes
  function togglePerformanceMode() {
    addActiveState(btnToggleMode);

    fetch("/api/toggle_mode")
      .then((response) => response.json())
      .then((data) => {
        const mode = data.mode;
        window.lightweightModeEnabled = mode === "lightweight";

        if (window.lightweightModeEnabled) {
          addAlert("Switched to lightweight mode for better performance");
          btnToggleMode.querySelector("span").textContent =
            "Switch to Full Mode";
          btnToggleMode.querySelector("i").className = "fas fa-tachometer-alt";
        } else {
          addAlert("Switched to full processing mode");
          btnToggleMode.querySelector("span").textContent =
            "Switch to Lightweight Mode";
          btnToggleMode.querySelector("i").className = "fas fa-bolt";
        }

        // Play a notification sound
        playSound("fallback_notification");
      })
      .catch((error) => {
        console.error("Error:", error);
        addAlert("Failed to toggle performance mode");
      })
      .finally(() => {
        removeActiveState(btnToggleMode);
      });
  }

  // Helper Functions

  // Add active state to button
  function addActiveState(button) {
    button.style.backgroundColor = "var(--primary-dark)";
  }

  // Remove active state from button
  function removeActiveState(button) {
    setTimeout(() => {
      button.style.backgroundColor = "";
    }, 300);
  }

  // Set up periodic updates
  simulatePeriodicUpdates();

  function simulatePeriodicUpdates() {
    // Periodically request updates from the server
    // This is just for demonstration purposes
    // In a real application, you'd use WebSockets or Server-Sent Events
    setInterval(() => {
      // Update connection status randomly (just for demonstration)
      const connectionStatus = document.getElementById("connection-status");
      if (Math.random() > 0.95) {
        if (connectionStatus.textContent === "Connected") {
          connectionStatus.textContent = "Reconnecting...";
          connectionStatus.style.color = "var(--warning)";
        } else {
          connectionStatus.textContent = "Connected";
          connectionStatus.style.color = "var(--success)";
        }
      }

      // Periodically check WebSocket connection and reconnect if needed
      if (socket && !socket.connected) {
        console.log("WebSocket disconnected, attempting to reconnect...");
        socket.connect();
      }
    }, 5000);
  }

  // Update connection status in UI
  function updateConnectionStatus(connected) {
    const connectionStatus = document.getElementById("connection-status");
    if (!connectionStatus) return;

    if (connected) {
      connectionStatus.textContent = "Connected";
      connectionStatus.style.color = "var(--success-color)";
    } else {
      connectionStatus.textContent = "Disconnected";
      connectionStatus.style.color = "var(--danger-color)";
    }
  }

  // Update microphone status with colored indicators
  function updateMicStatus(active) {
    const micStatus = document.getElementById("mic-status");

    if (active) {
      micStatus.textContent = "Voice: Active";
      micStatus.className = "text-success";
    } else {
      micStatus.textContent = "Voice: Inactive";
      micStatus.className = "";
    }
  }

  // Update system status
  function updateSystemStatus(status) {
    const systemStatus = document.getElementById("system-status");

    switch (status) {
      case "ok":
        systemStatus.textContent = "System: OK";
        systemStatus.className = "text-success";
        break;
      case "warning":
        systemStatus.textContent = "System: Warning";
        systemStatus.className = "text-warning";
        break;
      case "error":
        systemStatus.textContent = "System: Error";
        systemStatus.className = "text-danger";
        break;
      default:
        systemStatus.textContent = "System: Unknown";
        systemStatus.className = "";
    }
  }
});
