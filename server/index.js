const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const http = require("http");
const { Server } = require("socket.io");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files from frontend directory
app.use(express.static(path.join(__dirname, "../frontend")));

// Default route serves login
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/login.html"));
});

// Login route
app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/login.html"));
});

// Dashboard route
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/dashboard.html"));
});

// Navigate route
app.get('/navigate', (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/navigate.html"));
});

// Simulation route
app.get('/simulation', (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/simulation.html"));
});

// System route
app.get('/system', (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/system.html"));
});

const serverHttp = http.createServer(app);
const io = new Server(serverHttp, { cors: { origin: "*" } });

const upload = multer();

app.post("/api/detect", upload.single("file"), async (req, res) => {
  const formData = new FormData();
  formData.append("file", req.file.buffer, req.file.originalname);

  const response = await axios.post("http://localhost:5001/detect", formData, { headers: formData.getHeaders() });

  const detections = response.data;

  if (detections.detections.some(d => d.class === 7)) {
    io.emit("ambulance_detected", { route: ["INT-01","INT-02","INT-03"] });
  }

  res.json(detections);
});

app.post("/api/predict", async (req, res) => {
  const response = await axios.post("http://localhost:5002/predict", req.body);
  const pred = response.data;

  if (pred.predicted_density > 80) {
    io.emit("reroute_alert", { message: "High congestion!", route: ["INT-01","INT-04"] });
  }

  res.json(pred);
});

app.get("/api/simulate", async (req, res) => {
  const response = await axios.get("http://localhost:5003/simulate");
  res.json(response.data);
});
// Auth routes
app.post("/api/auth/login", (req, res) => {
  const { username, password } = req.body;
  if (username === "admin" && password === "admin") {
    res.json({ success: true });
  } else {
    res.status(401).json({ error: "Invalid credentials" });
  }
});

app.post("/api/auth/signup", (req, res) => {
  const { username, password, number_plate } = req.body;
  // Mock signup - always success
  res.json({ success: true });
});
serverHttp.listen(5000, () => console.log("Server running"));
