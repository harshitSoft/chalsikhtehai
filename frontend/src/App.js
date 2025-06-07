import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import Navbar from './components/Navbar';
import QRScanner from './components/QRScanner';
import QRGenerator from './components/QRGenerator';

// Create a theme instance
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navbar />
        <Container>
          <Routes>
            <Route path="/" element={
              <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                <h1>Welcome to Meter Reader</h1>
                <p>Scan QR codes to get user information</p>
              </div>
            } />
            <Route path="/scan" element={<QRScanner />} />
            <Route path="/generate" element={<QRGenerator />} />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

export default App;
